# GenAI - FastAPI LLM Backend (Module 1)

This small FastAPI app forwards a user `question` to a Hugging Face chat model and returns structured JSON including latency, token estimates, a simple intent/priority classification, and a few suggested support actions. It also records basic metrics to a `metrics.csv` file in the same folder.

## Quick Setup

1. Create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure environment variables in `Module 1/.env`:

```dotenv
# Hugging Face API token (required) - get from https://huggingface.co/settings/tokens
HF_API_KEY=hf_YOUR_REAL_API_KEY

# Default model to use if not provided in request
HF_DEFAULT_MODEL=openai/gpt-oss-20b:groq

# Cost estimation (per 1k tokens). Set to 0.0 if unknown.
COST_PER_1K_PROMPT=0.0
COST_PER_1K_COMPLETION=0.0

# Optional: override the Hugging Face router URL (usually not needed)
# HF_API_URL=https://router.huggingface.co/v1/chat/completions
```

3. Run the server locally from the `Module 1` folder:

```powershell
cd "Module 1"
uvicorn main:app --reload --port 8000
```

## API: /ask (POST)

Endpoint accepts JSON with the following shape:

```json
{ "question": "<your question>", "model": "<optional model id>" }
```

Example curl request:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"How to deploy a FastAPI app?"}'
```

## Response format (structured)

The API returns a JSON object with three top-level keys:

- `answer`: a dict containing `summary`, `intent`, and `priority`
  - `summary`: short 1–2 sentence summary of the generated answer
  - `intent`: a classification label (e.g. `incident`, `support`, `information`)
  - `priority`: `high`, `medium`, or `low`
- `confidence_score`: float between 0.0 and 1.0 indicating confidence in the `intent`
- `suggested_actions`: list of short action strings suitable for a support agent

Example response (trimmed):

```json
{
  "answer": { "summary": "Deploy using Uvicorn.", "intent": "support", "priority": "medium" },
  "confidence_score": 0.9,
  "suggested_actions": ["Provide step-by-step resolution guide", "Ask for environment and reproducible steps"]
}
```

## Metrics CSV

Each request appends a best-effort row to `metrics.csv` inside the `Module 1` folder. Columns include:
- `timestamp`: UTC ISO format timestamp
- `question`: the user's question
- `model`: model ID used
- `latency_ms`: round-trip latency in milliseconds
- `prompt_tokens`, `completion_tokens`, `total_tokens`: token counts
- `estimated_cost_usd`: estimated cost
- `intent`: inferred intent classification
- `priority`: assigned priority (high/medium/low)
- `confidence_score`: confidence in the intent classification

If writing to the CSV fails, it will not break the API response.

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_API_KEY` | Yes | — | Your Hugging Face API token |
| `HF_DEFAULT_MODEL` | No | `openai/gpt-oss-20b:groq` | Default model ID |
| `COST_PER_1K_PROMPT` | No | `0.0` | Cost per 1k prompt tokens |
| `COST_PER_1K_COMPLETION` | No | `0.0` | Cost per 1k completion tokens |
| `HF_API_URL` | No | `https://router.huggingface.co/v1/chat/completions` | API endpoint URL |

## Notes & Limitations

- Token counts are rough estimates (~1.3 tokens per word) when the upstream API does not return usage details.
- The intent classifier and priority heuristics are lightweight, rule-based, and intended for routing/support suggestions only.
- Make sure your `HF_API_KEY` has the needed permissions for the chosen model and your account has sufficient quota.
- The `.env` file should not be committed to version control — add it to `.gitignore`.

## Response Keys

- `question`: original question
- `answer`: generated text from Hugging Face model
- `model`: model ID used
- `latency_ms`: round-trip latency to Hugging Face (ms)
- `tokens`: object with `prompt_tokens`, `completion_tokens`, `total_tokens` (estimated via word count)
- `estimated_cost_usd`: cost estimate (default 0 for free tier)
- `raw_response`: raw Hugging Face API response

## Configuration

Environment variables:

- `HUGGINGFACE_API_KEY` (required): Your Hugging Face API token
- `HF_DEFAULT_MODEL`: Model ID (defaults to `meta-llama/Llama-2-7b-chat-hf`)
- `COST_PER_1K_PROMPT`: Cost per 1k prompt tokens (default 0.0)
- `COST_PER_1K_COMPLETION`: Cost per 1k completion tokens (default 0.0)

## Supported Models

Popular free/inference models on Hugging Face:

- `meta-llama/Llama-2-7b-chat-hf` (Llama 2 Chat, 7B)
- `mistralai/Mistral-7B-Instruct-v0.1` (Mistral, 7B)
- `tiiuae/falcon-7b-instruct` (Falcon, 7B)
- `google/flan-t5-base` (FLAN-T5, smaller)

Note: Token counts are **estimated** via word count (~1.3 tokens per word). The Hugging Face Inference API does not always return exact token counts in the response.
