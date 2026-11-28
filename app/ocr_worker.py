"""Private subprocess entry point for local OCR; report bytes arrive on stdin."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "OCR worker expects a report filename."}))
        return 2

    try:
        from app.services.report_extraction import extract_report

        content = sys.stdin.buffer.read()
        # RapidOCR may emit informational logs. Keep stdout reserved for JSON.
        with redirect_stdout(sys.stderr):
            _text, candidates = extract_report(sys.argv[1], content)
        print(json.dumps({"candidates": [asdict(candidate) for candidate in candidates]}))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
