import sys
import platform

def greet(name: str = "World") -> str:
	return f"Hello, {name}!"


def main() -> int:
	print(greet())
	print("Python:", sys.version.split()[0], "Platform:", platform.system())
	print("2+2 =", 2 + 2)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
