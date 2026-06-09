import json
from pathlib import Path
from statistics import mean
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BENCHMARK_FILE = DATA_DIR / "benchmark_events.jsonl"
DLQ_FILE = DATA_DIR / "dlq.jsonl"
OUTPUT_FILE = DATA_DIR / "benchmark_summary.json"


def load_jsonl(path: Path):
    """
    Load a JSONL file safely.
    Invalid JSON lines are ignored so one corrupted log line does not break the analyzer.
    """
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def parse_iso_datetime(value):
    """
    Parse ISO datetime strings such as:
    2026-06-06T15:05:23.539222+00:00
    """
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def calculate_time_diff_ms(start_value, end_value):
    """
    Calculate millisecond difference between two ISO datetime strings.
    """
    start_time = parse_iso_datetime(start_value)
    end_time = parse_iso_datetime(end_value)

    if start_time is None or end_time is None:
        return None

    return (end_time - start_time).total_seconds() * 1000


def get_nested_value(data, path, default=None):
    """
    Safely read nested dictionary values.

    Example:
    get_nested_value(event, ["local_model", "local_latency_ms"])
    """
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def clean_number_list(values):
    """
    Keep only int/float values.
    """
    return [
        value for value in values
        if isinstance(value, (int, float)) and value >= 0
    ]


def safe_mean(values):
    values = clean_number_list(values)

    if not values:
        return None

    return mean(values)


def percentile(values, p):
    """
    Simple percentile calculation for benchmark reporting.
    """
    values = clean_number_list(values)

    if not values:
        return None

    sorted_values = sorted(values)
    index = int(round((p / 100) * (len(sorted_values) - 1)))

    return sorted_values[index]


def percentage(numerator, denominator):
    if denominator == 0:
        return None

    return (numerator / denominator) * 100


def analyse_benchmark():
    benchmark_events = load_jsonl(BENCHMARK_FILE)
    dlq_events = load_jsonl(DLQ_FILE)

    local_events = [
        event for event in benchmark_events
        if event.get("event_type") == "local_pipeline_result"
    ]

    llm_events = [
        event for event in benchmark_events
        if event.get("event_type") == "llm_escalation_result"
    ]

    total_valid_requests = len(local_events)
    total_invalid_requests = len(dlq_events)
    total_requests = total_valid_requests + total_invalid_requests

    # -----------------------------
    # Local pipeline latency
    # -----------------------------
    # Your current log does not directly store total_pipeline_latency_ms.
    # It stores received_at and processed_at, so we calculate it here.
    total_pipeline_latencies = [
        calculate_time_diff_ms(
            event.get("received_at"),
            event.get("processed_at")
        )
        for event in local_events
    ]

    total_pipeline_latencies = clean_number_list(total_pipeline_latencies)

    # Your current log stores local model latency here:
    # event["local_model"]["local_latency_ms"]
    local_model_latencies = [
        get_nested_value(event, ["local_model", "local_latency_ms"])
        for event in local_events
    ]

    local_model_latencies = clean_number_list(local_model_latencies)

    # -----------------------------
    # Local decision statistics
    # -----------------------------
    local_only_count = 0
    escalation_requested_count = 0

    decision_reasons = {}

    for event in local_events:
        decision = event.get("decision", {})
        trigger_reason = decision.get("trigger_reason", "UNKNOWN")

        decision_reasons[trigger_reason] = decision_reasons.get(trigger_reason, 0) + 1

        if decision.get("escalate_to_llm") is True:
            escalation_requested_count += 1
        else:
            local_only_count += 1

    # -----------------------------
    # LLM escalation statistics
    # -----------------------------
    llm_escalation_event_count = len(llm_events)

    llm_success_count = 0
    llm_failure_count = 0
    llm_timeout_count = 0
    llm_cooldown_count = 0
    llm_skipped_count = 0
    actual_cloud_llm_call_count = 0

    llm_status_counts = {}

    llm_background_elapsed_latencies = []
    actual_cloud_llm_latencies = []

    for event in llm_events:
        llm = event.get("llm", {})

        status = llm.get("llm_status") or llm.get("status") or "unknown"
        llm_called = llm.get("llm_called") is True

        llm_status_counts[status] = llm_status_counts.get(status, 0) + 1

        if llm_called:
            actual_cloud_llm_call_count += 1

        if status == "success":
            llm_success_count += 1
        elif status == "cooldown":
            llm_cooldown_count += 1
        elif status == "skipped":
            llm_skipped_count += 1
        else:
            llm_failure_count += 1

        if status == "timeout":
            llm_timeout_count += 1

        # Background task elapsed time
        background_elapsed_ms = calculate_time_diff_ms(
            event.get("started_at"),
            event.get("completed_at")
        )

        if isinstance(background_elapsed_ms, (int, float)) and background_elapsed_ms >= 0:
            llm_background_elapsed_latencies.append(background_elapsed_ms)

        # Actual cloud latency should only count real LLM calls.
        # In your log, cooldown events have llm_called=False and should not be counted as API latency.
        if llm_called:
            actual_cloud_latency_ms = background_elapsed_ms

            if actual_cloud_latency_ms is None:
                actual_cloud_latency_ms = llm.get("llm_latency_ms")

            if isinstance(actual_cloud_latency_ms, (int, float)) and actual_cloud_latency_ms >= 0:
                actual_cloud_llm_latencies.append(actual_cloud_latency_ms)

    # -----------------------------
    # DLQ / validation statistics
    # -----------------------------
    invalid_payload_rate = percentage(total_invalid_requests, total_requests)

    # Since this analyzer only reads rejected records that already reached the DLQ,
    # the interception rate is treated as 100% for logged invalid payloads.
    dirty_data_interception_rate = 100.0 if total_invalid_requests > 0 else None

    # -----------------------------
    # Estimated cost model
    # -----------------------------
    # This is only an estimated cost model.
    # Replace this value later if you use real Gemini token pricing.
    estimated_cost_per_cloud_llm_call_usd = 0.001

    always_call_cloud_cost = (
        total_valid_requests * estimated_cost_per_cloud_llm_call_usd
    )

    decision_layer_escalation_cost = (
        llm_escalation_event_count * estimated_cost_per_cloud_llm_call_usd
    )

    actual_cloud_call_cost = (
        actual_cloud_llm_call_count * estimated_cost_per_cloud_llm_call_usd
    )

    decision_layer_cost_saving = None
    actual_cloud_cost_saving = None

    if always_call_cloud_cost > 0:
        decision_layer_cost_saving = (
            (always_call_cloud_cost - decision_layer_escalation_cost)
            / always_call_cloud_cost
        ) * 100

        actual_cloud_cost_saving = (
            (always_call_cloud_cost - actual_cloud_call_cost)
            / always_call_cloud_cost
        ) * 100

    result = {
        "request_summary": {
            "total_requests": total_requests,
            "valid_requests": total_valid_requests,
            "invalid_requests_dlq": total_invalid_requests,
            "invalid_payload_rate_percent": invalid_payload_rate,
            "dirty_data_interception_rate_percent": dirty_data_interception_rate,
        },
        "local_pipeline": {
            "local_only_count": local_only_count,
            "local_only_rate_percent": percentage(local_only_count, total_valid_requests),
            "escalation_requested_count": escalation_requested_count,
            "escalation_requested_rate_percent": percentage(
                escalation_requested_count,
                total_valid_requests
            ),
            "decision_reasons": decision_reasons,
            "average_total_pipeline_latency_ms": safe_mean(total_pipeline_latencies),
            "p95_total_pipeline_latency_ms": percentile(total_pipeline_latencies, 95),
            "p99_total_pipeline_latency_ms": percentile(total_pipeline_latencies, 99),
            "average_local_model_latency_ms": safe_mean(local_model_latencies),
            "p95_local_model_latency_ms": percentile(local_model_latencies, 95),
            "p99_local_model_latency_ms": percentile(local_model_latencies, 99),
        },
        "llm_escalation": {
            "llm_escalation_event_count": llm_escalation_event_count,
            "llm_escalation_event_rate_percent": percentage(
                llm_escalation_event_count,
                total_valid_requests
            ),
            "actual_cloud_llm_call_count": actual_cloud_llm_call_count,
            "actual_cloud_llm_call_rate_percent": percentage(
                actual_cloud_llm_call_count,
                total_valid_requests
            ),
            "llm_success_count": llm_success_count,
            "llm_cooldown_count": llm_cooldown_count,
            "llm_skipped_count": llm_skipped_count,
            "llm_failure_count": llm_failure_count,
            "llm_timeout_count": llm_timeout_count,
            "llm_status_counts": llm_status_counts,
            "average_llm_background_elapsed_ms": safe_mean(
                llm_background_elapsed_latencies
            ),
            "p95_llm_background_elapsed_ms": percentile(
                llm_background_elapsed_latencies,
                95
            ),
            "p99_llm_background_elapsed_ms": percentile(
                llm_background_elapsed_latencies,
                99
            ),
            "average_actual_cloud_llm_latency_ms": safe_mean(
                actual_cloud_llm_latencies
            ),
            "p95_actual_cloud_llm_latency_ms": percentile(
                actual_cloud_llm_latencies,
                95
            ),
            "p99_actual_cloud_llm_latency_ms": percentile(
                actual_cloud_llm_latencies,
                99
            ),
        },
        "estimated_cost": {
            "estimated_cost_per_cloud_llm_call_usd": estimated_cost_per_cloud_llm_call_usd,
            "always_call_cloud_cost_usd": always_call_cloud_cost,
            "decision_layer_escalation_cost_usd": decision_layer_escalation_cost,
            "actual_cloud_call_cost_usd": actual_cloud_call_cost,
            "decision_layer_estimated_cost_saving_percent": decision_layer_cost_saving,
            "actual_cloud_call_estimated_cost_saving_percent": actual_cloud_cost_saving,
        },
    }

    return result


def print_section(title, data):
    print(f"\n[{title}]")

    for key, value in data.items():
        print(f"{key}: {value}")


def print_report(result):
    print("\n=== Benchmark Analysis Report ===")

    print_section("Request Summary", result["request_summary"])
    print_section("Local Pipeline", result["local_pipeline"])
    print_section("LLM Escalation", result["llm_escalation"])
    print_section("Estimated Cost", result["estimated_cost"])


def save_report(result):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"\nSaved benchmark summary to: {OUTPUT_FILE}")



if __name__ == "__main__":
    analysis_result = analyse_benchmark()
    print_report(analysis_result)
    save_report(analysis_result)
