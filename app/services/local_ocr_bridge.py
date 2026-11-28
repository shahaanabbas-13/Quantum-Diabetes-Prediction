"""Run optional document OCR in its own Python 3.12 process in memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from app.services.report_extraction import Candidate


def _worker_python() -> Path:
    configured = os.environ.get("QD_OCR_PYTHON")
    if configured:
        executable = Path(configured).expanduser()
    else:
        repository = Path(__file__).resolve().parents[2]
        executable = repository / ".venv-documents" / "Scripts" / "python.exe"
        if not executable.is_file() and sys.version_info >= (3, 12):
            executable = Path(sys.executable)
    if not executable.is_file():
        raise RuntimeError(
            "Local OCR runtime not found. Create .venv-documents with Python 3.12 "
            "and install app/requirements-documents.txt, or set QD_OCR_PYTHON."
        )
    return executable


def extract_candidates_locally(file_name: str, content: bytes) -> list[Candidate]:
    """Send report bytes over a local process pipe; no report file is created."""
    repository = Path(__file__).resolve().parents[2]
    command = [
        str(_worker_python()),
        "-m",
        "app.ocr_worker",
        file_name,
    ]
    try:
        completed = subprocess.run(
            command,
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repository,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start the local OCR runtime: {exc}") from exc

    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            message = payload.get("error", message)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise RuntimeError(message or "Local OCR extraction failed.")

    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        return [Candidate(**item) for item in payload["candidates"]]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Local OCR returned an invalid response.") from exc
