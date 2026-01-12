import os
import time
import csv
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None


app = FastAPI(title="GenAI - Hugging Face LLM Metrics API")

# Configure Hugging Face API key and defaults from environment
HF_API_KEY = os.getenv("HF_API_KEY", "")
if not HF_API_KEY or HF_API_KEY == "hf_REPLACE_WITH_YOUR_API_KEY":
    print("⚠️  WARNING: HF_API_KEY not set or still has placeholder value in .env file")
    print("   Set a real Hugging Face API token in Module 1/.env before running requests.")

DEFAULT_MODEL = os.getenv("HF_DEFAULT_MODEL", "openai/gpt-oss-20b:groq")

# Cost estimation (per 1k tokens). Set via env vars to reflect your pricing.
# Note: Hugging Face Inference API pricing varies; adjust based on your plan.
COST_PER_1K_PROMPT = float(os.getenv("COST_PER_1K_PROMPT", "0.0"))
COST_PER_1K_COMPLETION = float(os.getenv("COST_PER_1K_COMPLETION", "0.0"))

# Hugging Face Inference API endpoint (configurable via .env)
HF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cost = (prompt_tokens / 1000.0) * COST_PER_1K_PROMPT + (completion_tokens / 1000.0) * COST_PER_1K_COMPLETION
    return float(cost)


def estimate_tokens(text: str) -> int:
    # Very rough approximation: ~1.3 tokens per word
    words = len(text.split())
    return max(1, int(words * 1.3))


def _metrics_csv_path() -> str:
    return os.path.join(os.path.dirname(__file__), "metrics.csv")


def save_metrics_csv(entry: Dict[str, Any]) -> None:
    path = _metrics_csv_path()
    fieldnames = [
        "timestamp",
        "question",
        "model",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "intent",
        "priority",
        "confidence_score",
    ]
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": entry.get("question", ""),
                "model": entry.get("model", ""),
                "latency_ms": entry.get("latency_ms", ""),
                "prompt_tokens": entry.get("tokens", {}).get("prompt_tokens", ""),
                "completion_tokens": entry.get("tokens", {}).get("completion_tokens", ""),
                "total_tokens": entry.get("tokens", {}).get("total_tokens", ""),
                "estimated_cost_usd": entry.get("estimated_cost_usd", ""),
                "intent": entry.get("intent", ""),
                "priority": entry.get("priority", ""),
                "confidence_score": entry.get("confidence_score", ""),
            }
            writer.writerow(row)
    except Exception as e:
        print("Warning: failed to write metrics CSV:", e)

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY not set in environment")

    model = req.model or DEFAULT_MODEL

    # Determine token for router: prefer HF_TOKEN, fall back to HUGGINGFACE_API_KEY
    token = HF_API_KEY
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN or HUGGINGFACE_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print("Question:", req.question, "Model:", model, "Headers:", headers)
    # Router expects OpenAI-like chat completions payload
    payload = {
        "messages": [
            {"role": "user", "content": req.question}
        ],
        "model": model
    }

    start = time.time()
    try:
        resp = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face router request failed: {e}")
    latency_ms = (time.time() - start) * 1000.0

    # Parse response similar to OpenAI router-style response
    answer = ""
    try:
        # expected shape: { choices: [ { message: { content: "..." } } ], usage: {...} }
        answer = data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        # fallback: try other common fields
        if isinstance(data, list) and len(data) > 0:
            answer = data[0].get("generated_text", "")
        else:
            answer = str(data)

    # Try to read usage if present
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens_info = {}
    if usage and all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        tokens_info = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    else:
        # fallback estimate
        prompt_tokens = estimate_tokens(req.question)
        completion_tokens = estimate_tokens(answer)
        tokens_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "note": "estimated via word count (rough approximation)",
        }

    estimated_cost = estimate_cost(tokens_info["prompt_tokens"], tokens_info["completion_tokens"])

    # Build structured response expected by the caller
    def _extract_summary(text: str) -> str:
        if not text:
            return ""
        # naive sentence splitter
        for sep in ['. ', '? ', '! ']:
            if sep in text:
                parts = text.split(sep)
                first = parts[0].strip()
                # return first sentence, appended with a period if missing
                return first + ('.' if not first.endswith(('.', '!', '?')) else '')
        # fallback: return up to 160 chars
        return (text[:160] + '...') if len(text) > 160 else text

    def _classify_intent(question: str, answer_text: str) -> str:
        q = (question or "").lower()
        keywords_high = ["urgent", "asap", "immediately", "error", "fail", "down", "outage"]
        keywords_medium = ["how", "configure", "setup", "help", "install", "deploy", "guide", "performance"]
        keywords_low = ["idea", "recommend", "suggest", "info", "what is", "explain"]

        for k in keywords_high:
            if k in q:
                return "incident"
        for k in keywords_medium:
            if k in q:
                return "support"
        for k in keywords_low:
            if k in q:
                return "information"
        # fallback: infer from answer
        a = (answer_text or "").lower()
        if any(x in a for x in ["error", "stacktrace", "exception"]):
            return "incident"
        if len(a) < 50:
            return "information"
        return "support"

    def _priority_from_intent(intent_label: str, question: str) -> str:
        if intent_label == "incident":
            return "high"
        if intent_label == "support":
            # if question contains "urgent" escalate
            if any(x in (question or "").lower() for x in ["urgent", "now", "asap"]):
                return "high"
            return "medium"
        return "low"

    def _confidence(answer_text: str, intent_label: str) -> float:
        score = 0.5
        if answer_text:
            score += 0.25
        if intent_label in ("incident", "support"):
            score += 0.15
        return min(1.0, round(score, 2))

    def _suggest_actions(intent_label: str) -> list:
        if intent_label == "incident":
            return [
                "Acknowledge incident and open ticket",
                "Request logs and timestamps from user",
                "Escalate to on-call engineer",
            ]
        if intent_label == "support":
            return [
                "Provide step-by-step resolution guide",
                "Ask for environment and reproducible steps",
                "Offer follow-up troubleshooting session",
            ]
        return [
            "Provide documentation links",
            "Offer examples and further reading",
        ]

    structured_answer = {
        "summary": _extract_summary(answer),
        "intent": _classify_intent(req.question, answer),
        "priority": "medium",
    }

    structured_answer["priority"] = _priority_from_intent(structured_answer["intent"], req.question)

    confidence = _confidence(answer, structured_answer["intent"])
    actions = _suggest_actions(structured_answer["intent"])

    response = {
        "answer": structured_answer,
        "confidence_score": float(confidence),
        "suggested_actions": actions,
    }

    # Persist basic metrics to CSV (best-effort)
    try:
        metrics_entry = {
            "question": req.question,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens_info,
            "estimated_cost_usd": round(estimated_cost, 6),
            "intent": structured_answer.get("intent"),
            "priority": structured_answer.get("priority"),
            "confidence_score": confidence,
        }
        save_metrics_csv(metrics_entry)
    except Exception:
        # do not fail the request if metrics saving fails
        pass

    return response