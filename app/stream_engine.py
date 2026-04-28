import asyncio
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import ValidationError
from gateway import TrafficRecord

# Standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "PEMS03.npz"
DLQ_LOG = BASE_DIR / "data" / "dlq_intercept.json"  # 将日志改存到 data 文件夹


async def simulate_iot_sensor_stream():
    """Producer: Extracts real-time slices from PEMS03 matrix."""
    if not DATA_FILE.exists():
        logger.error(f"Dataset missing at {DATA_FILE}")
        return

    traffic_matrix = np.load(DATA_FILE)['data']
    current_time = datetime(2026, 4, 28, 8, 0, 0)

    # We keep the range(20) as our current simulation window
    for t in range(20):
        for sensor_idx in [0, 1]:
            features = traffic_matrix[t, sensor_idx]

            # Data Imputation & Anomaly Injection for testing
            flow_val = float(features[0])
            speed_val = 60.0 + np.random.normal(0, 5)
            if np.random.rand() < 0.1: speed_val = -99.0  # Synthetic Fault

            yield {
                "sensor_id": f"S-{sensor_idx:03d}",
                "timestamp": current_time.isoformat(),
                "flow": round(flow_val, 2),
                "speed": round(speed_val, 2)
            }
        current_time += timedelta(minutes=5)
        await asyncio.sleep(0.1)


async def stream_processor():
    """Consumer: Validates stream and manages DLQ."""
    clean_queue = asyncio.Queue()
    dlq_list = []

    async for record in simulate_iot_sensor_stream():
        try:
            # Task 1: Pydantic Validation
            valid_data = TrafficRecord(**record)
            await clean_queue.put(valid_data.model_dump())
            logger.info(f"[PASS] {record['timestamp']} - {record['sensor_id']}")

        except ValidationError as e:
            # Task 2: DLQ Interception
            fault_entry = {"data": record, "reason": e.errors()[0]['msg']}
            dlq_list.append(fault_entry)
            logger.warning(f"[DLQ] {record['timestamp']} - {record['sensor_id']} - Reason: {fault_entry['reason']}")

    # Task 3: Persistence (Keeping the .json for evidence)
    if dlq_list:
        with open(DLQ_LOG, "w") as f:
            json.dump(dlq_list, f, indent=4)
        logger.info(f"DLQ records persisted to {DLQ_LOG}")


if __name__ == "__main__":
    try:
        asyncio.run(stream_processor())
    except KeyboardInterrupt:
        logger.info("Stream Engine stopped by user.")