import argparse
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "PEMS07.npz"

DEFAULT_API_URL = "http://127.0.0.1:8000/v1/traffic/events"


def load_pems_data():
    """
    PEMS03.npz contains traffic flow data.
    Expected shape is usually: [time_steps, sensors, features].
    In this project, feature 0 is used as traffic flow.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"PEMS03 dataset not found: {DATA_FILE}")

    data = np.load(DATA_FILE)["data"]
    print(f"Loaded PEMS03 data shape: {data.shape}")

    return data


def build_pems_payload(
    traffic_matrix,
    time_index: int,
    sensor_index: int,
    current_time: datetime,
) -> dict:
    flow_value = float(traffic_matrix[time_index, sensor_index, 0])

    return {
        "sensor_id": f"S-{sensor_index:03d}",
        "timestamp": current_time.strftime("%H:%M:%S"),
        "flow": round(flow_value, 2),
        # PEMS03 mainly provides flow. Speed and occupancy are simulated
        # supplementary fields for validation testing.
        "speed": round(random.uniform(45, 85), 2),
        "occupancy": round(random.uniform(10, 70), 2),
    }


def inject_dirty_payload(payload: dict) -> tuple[dict, str]:
    """
    Create controlled dirty data to test the Pydantic gateway and DLQ.
    Returns both the dirty payload and the dirty case label.
    """
    dirty_case = random.choice(
        [
            "negative_speed",
            "empty_sensor_id",
            "negative_flow",
            "bad_occupancy",
            "bad_timestamp",
            "numeric_string_flow",
            "extra_field",
        ]
    )

    dirty_payload = dict(payload)

    if dirty_case == "negative_speed":
        dirty_payload["speed"] = -99

    elif dirty_case == "empty_sensor_id":
        dirty_payload["sensor_id"] = ""

    elif dirty_case == "negative_flow":
        dirty_payload["flow"] = -10

    elif dirty_case == "bad_occupancy":
        dirty_payload["occupancy"] = 150

    elif dirty_case == "bad_timestamp":
        dirty_payload["timestamp"] = "not-a-time"

    elif dirty_case == "numeric_string_flow":
        dirty_payload["flow"] = str(dirty_payload["flow"])

    elif dirty_case == "extra_field":
        dirty_payload["unexpected_field"] = "should_be_rejected"

    return dirty_payload, dirty_case


def inject_accident_scenario(
    payload: dict,
    time_index: int,
    sensor_index: int,
    accident_enabled: bool,
) -> tuple[dict, bool]:
    """
    Controlled anomaly scenario:
    Sharply reduce flow for sensor S-000 at selected time steps,
    so the Decision Layer can be tested quickly.
    """
    if not accident_enabled:
        return payload, False

    accident_payload = dict(payload)

    if time_index in [20, 21] and sensor_index == 0:
        accident_payload["flow"] = max(
            1.0,
            round(accident_payload["flow"] * 0.1, 2),
        )
        return accident_payload, True

    return payload, False


def send_payload(api_url: str, payload: dict, timeout: float = 15.0) -> tuple[int | None, object]:

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)

        try:
            body = response.json()
        except Exception:
            body = response.text

        return response.status_code, body

    except requests.RequestException as exc:
        return None, str(exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PEMS03 traffic sensor client for FastAPI benchmark testing."
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_API_URL,
        help="FastAPI traffic ingestion endpoint.",
    )

    parser.add_argument(
        "--dirty-ratio",
        type=float,
        default=0.10,
        help="Probability of injecting a dirty payload. Example: 0.10 means 10%%.",
    )

    parser.add_argument(
        "--sensors",
        type=int,
        default=2,
        help="Number of sensors to simulate from PEMS03.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=35,
        help="Number of time steps to send.",
    )

    parser.add_argument(
        "--time-start",
        type=int,
        default=10,
        help="Starting time index in the PEMS03 dataset.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sleep interval in seconds after each time step.",
    )

    parser.add_argument(
        "--accident",
        action="store_true",
        help="Enable controlled traffic accident scenario injection.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dirty-data injection.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-request output.",
    )

    return parser.parse_args()


def validate_args(args, traffic_matrix):
    if not 0 <= args.dirty_ratio <= 1:
        raise ValueError("--dirty-ratio must be between 0 and 1")

    if args.sensors <= 0:
        raise ValueError("--sensors must be greater than 0")

    if args.sensors > traffic_matrix.shape[1]:
        raise ValueError(
            f"--sensors={args.sensors} is too large. "
            f"PEMS03 only has {traffic_matrix.shape[1]} sensors."
        )

    if args.duration <= 0:
        raise ValueError("--duration must be greater than 0")

    if args.time_start < 0:
        raise ValueError("--time-start must be greater than or equal to 0")

    if args.time_start + args.duration >= traffic_matrix.shape[0]:
        raise ValueError(
            f"time range exceeds dataset length. "
            f"time_start={args.time_start}, duration={args.duration}, "
            f"dataset time steps={traffic_matrix.shape[0]}"
        )

    if args.interval < 0:
        raise ValueError("--interval must be greater than or equal to 0")


def main():
    args = parse_args()
    random.seed(args.seed)

    traffic_matrix = load_pems_data()
    validate_args(args, traffic_matrix)

    start_time = datetime(2026, 4, 28, 8, 0, 0)

    sensor_indices = list(range(args.sensors))
    time_end = args.time_start + args.duration

    print("\nStarting PEMS03 external traffic sensor client...")
    print(f"API URL: {args.url}")
    print(f"Time range: {args.time_start} -> {time_end - 1}")
    print(f"Sensors: {sensor_indices}")
    print(f"Dirty ratio: {args.dirty_ratio}")
    print(f"Interval: {args.interval} seconds")
    print(f"Accident scenario enabled: {args.accident}")
    print("-" * 80)

    current_time = start_time

    total_sent = 0
    valid_sent = 0
    dirty_sent = 0
    accident_sent = 0
    success_responses = 0
    rejected_responses = 0
    failed_requests = 0

    dirty_case_counts = {}

    started_at = time.perf_counter()

    for time_index in range(args.time_start, time_end):
        for sensor_idx in sensor_indices:
            payload = build_pems_payload(
                traffic_matrix=traffic_matrix,
                time_index=time_index,
                sensor_index=sensor_idx,
                current_time=current_time,
            )

            payload, accident_injected = inject_accident_scenario(
                payload=payload,
                time_index=time_index,
                sensor_index=sensor_idx,
                accident_enabled=args.accident,
            )

            payload_type = "normal"
            dirty_case = None

            if random.random() < args.dirty_ratio:
                payload, dirty_case = inject_dirty_payload(payload)
                payload_type = "dirty"
                dirty_sent += 1
                dirty_case_counts[dirty_case] = dirty_case_counts.get(dirty_case, 0) + 1
            else:
                valid_sent += 1

            if accident_injected:
                accident_sent += 1

            total_sent += 1

            status_code, response_body = send_payload(args.url, payload)

            if status_code == 200:
                success_responses += 1
            elif status_code in [400, 422]:
                rejected_responses += 1
            else:
                failed_requests += 1

            if not args.quiet:
                print(f"Payload type: {payload_type}")
                if dirty_case:
                    print(f"Dirty case: {dirty_case}")
                if accident_injected:
                    print("Accident scenario injected")
                print("Payload:", payload)
                print("Status:", status_code)
                print("Response:", response_body)
                print("-" * 80)

        current_time += timedelta(minutes=5)

        if args.interval > 0:
            time.sleep(args.interval)

    elapsed = time.perf_counter() - started_at

    print("\nPEMS03 client run completed.")
    print("=" * 80)
    print(f"Total sent: {total_sent}")
    print(f"Expected valid payloads sent: {valid_sent}")
    print(f"Expected dirty payloads sent: {dirty_sent}")
    print(f"Accident scenario payloads sent: {accident_sent}")
    print(f"HTTP 200 accepted responses: {success_responses}")
    print(f"HTTP 400/422 rejected responses: {rejected_responses}")
    print(f"Failed requests: {failed_requests}")
    print(f"Dirty case counts: {dirty_case_counts}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    if elapsed > 0:
        print(f"Approximate send rate: {total_sent / elapsed:.2f} requests/second")


if __name__ == "__main__":
    main()
