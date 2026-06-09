import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np

from app.gateway import TrafficRecord


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "app" / "traffic_model.txt"


class LocalModelService:
    """
    Track A: local LightGBM inference service.

    The model was trained with a sliding window of 3 previous flow values.
    Therefore, each sensor needs at least 3 historical flow records before prediction.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.model = lgb.Booster(model_file=str(MODEL_FILE))
        self.sensor_memory = defaultdict(lambda: deque(maxlen=self.window_size))

    def predict(self, event: TrafficRecord) -> dict[str, Any]:
        start = time.perf_counter()

        sensor_id = event.sensor_id
        actual_flow = float(event.flow)

        history = self.sensor_memory[sensor_id]

        # Not enough history yet, so the local model cannot predict.
        if len(history) < self.window_size:
            history.append(actual_flow)

            latency_ms = round((time.perf_counter() - start) * 1000, 3)

            return {
                "model_status": "warmup",
                "sensor_id": sensor_id,
                "actual_flow": actual_flow,
                "predicted_flow": None,
                "flow_drop": None,
                "history": list(history),
                "local_latency_ms": latency_ms,
                "message": f"Waiting for {self.window_size} historical points before prediction."
            }

        # LightGBM expects shape: [1, 3]
        features = np.array(list(history)).reshape(1, -1)
        predicted_flow = float(self.model.predict(features)[0])

        flow_drop = predicted_flow - actual_flow

        # Update memory after prediction
        history.append(actual_flow)

        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        return {
            "model_status": "predicted",
            "sensor_id": sensor_id,
            "actual_flow": round(actual_flow, 3),
            "predicted_flow": round(predicted_flow, 3),
            "flow_drop": round(flow_drop, 3),
            "history": list(history),
            "local_latency_ms": latency_ms
        }


local_model_service = LocalModelService()