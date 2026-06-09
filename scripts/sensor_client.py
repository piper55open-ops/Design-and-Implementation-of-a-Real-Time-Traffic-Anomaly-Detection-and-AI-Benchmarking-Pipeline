import time
import random
import requests
from datetime import datetime


API_URL = "http://127.0.0.1:8000/v1/traffic/events"


def build_normal_payload(index: int) -> dict:
    return {
        "sensor_id": f"S-{index:03d}",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "flow": round(random.uniform(80, 160), 2),
        "speed": round(random.uniform(40, 90), 2),
        "occupancy": round(random.uniform(10, 70), 2)
    }


def build_dirty_payload() -> dict:
    dirty_cases = [
        {
            "sensor_id": "S-BAD-001",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "flow": 120.5,
            "speed": -99,
            "occupancy": 30.0
        },
        {
            "sensor_id": "",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "flow": 120.5,
            "speed": 60.0,
            "occupancy": 30.0
        },
        {
            "sensor_id": "S-BAD-003",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "flow": -10,
            "speed": 60.0,
            "occupancy": 30.0
        },
        {
            "sensor_id": "S-BAD-004",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "flow": 100.0,
            "speed": 60.0,
            "occupancy": 150
        }
    ]

    return random.choice(dirty_cases)


def send_payload(payload: dict) -> None:
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        print("Status:", response.status_code)
        print("Response:", response.json())
        print("-" * 80)
    except requests.RequestException as exc:
        print("Request failed:", exc)


def main():
    print("Starting external traffic sensor client...")

    for i in range(10):
        if i in [3, 7]:
            payload = build_dirty_payload()
            print("Sending dirty payload:")
        else:
            payload = build_normal_payload(i)
            print("Sending normal payload:")

        print(payload)
        send_payload(payload)
        time.sleep(1)


if __name__ == "__main__":
    main()