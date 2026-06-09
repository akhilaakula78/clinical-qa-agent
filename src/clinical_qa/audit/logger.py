"""JSONL audit log. Each agent step → one line."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from clinical_qa.config import settings

_lock = Lock()
_session_id = str(uuid.uuid4())


def set_session(sid: str) -> None:
    global _session_id
    _session_id = sid


def log_event(step: str, detail: dict[str, Any]) -> None:
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": _session_id,
        "step": step,
        "detail": detail,
    }
    line = json.dumps(rec, default=str)
    with _lock, open(path, "a") as f:
        f.write(line + "\n")
