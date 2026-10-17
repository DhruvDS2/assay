from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load `.env` from the repo root into os.environ on import.

    Zero-dependency on purpose: the code reads secrets straight from the
    process environment (the anthropic SDK and app/main.py both do), so a
    `.env` file is otherwise inert. Existing environment variables win — an
    explicit `export ANTHROPIC_API_KEY=...` always overrides the file.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
