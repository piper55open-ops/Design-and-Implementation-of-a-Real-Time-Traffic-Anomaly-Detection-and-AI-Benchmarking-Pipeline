import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_LOG_FILE = BASE_DIR / "data" / "benchmark_events.jsonl"


def write_benchmark_event(record: dict[str, Any]) -> None:
    """
    Write one accepted traffic event into benchmark_events.jsonl.

    This file will later be used for:
    - dashboard
    - latency analysis
    - LLM trigger rate
    - cost analysis
    - final report evidence
    """
    BENCHMARK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record.setdefault("logged_at", datetime.now(timezone.utc).isoformat())

    with BENCHMARK_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")