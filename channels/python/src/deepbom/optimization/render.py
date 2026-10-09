"""Read-only PDF renderer for validated report JSON supplied through stdin."""
import json
import sys
from .._native.core import engine
from .report import _pdf


def main():
    payload = sys.stdin.buffer.read(16 * 1024 * 1024 + 1)
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Report exceeds the 16 MiB import limit")
    # Validate via the same browser/CLI owner, not a Python interpretation.
    data = engine("optimization_import", {"text": payload.decode("utf-8")})
    sys.stdout.buffer.write(_pdf(data["report"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Report rendering failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
