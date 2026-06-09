import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DLQ_FILE = BASE_DIR / "data" / "dlq.jsonl"


def write_dlq(
    request_id: str,
    payload: Any,
    error_type: str,
    error_reason: Any,
    source: str
) -> None:
    DLQ_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "error_type": error_type,
        "error_reason": error_reason,
        "payload": payload
    }

    with DLQ_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")