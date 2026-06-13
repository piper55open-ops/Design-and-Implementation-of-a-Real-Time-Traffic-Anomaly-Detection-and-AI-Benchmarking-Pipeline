import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from app.gateway import TrafficRecord


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "app" / "traffic_model.txt"

STEPS_PER_DAY = 288  # 一天 288 个 5 分钟
WINDOW_SIZE = 12     # 新模型需要最近 12 个历史点


class LocalModelService:
    """
    Local LightGBM inference service.

    New model:
    - uses 12 historical flow values
    - builds 27 features
    - predicts next-step traffic flow
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.model = lgb.Booster(model_file=str(MODEL_FILE))

        # 每个 sensor 保存最近 12 个历史流量
        self.sensor_memory = defaultdict(lambda: deque(maxlen=self.window_size))

    def _get_current_step(self) -> int:
        """
        当前是一天中的第几个 5 分钟。
        例如 00:00 -> 0, 00:05 -> 1, 01:00 -> 12
        """
        now = time.localtime()
        return now.tm_hour * 12 + now.tm_min // 5

    def _build_features(self, history_values: list[float]) -> tuple[np.ndarray, float, float]:
        """
        把最近 12 个历史流量值变成新模型需要的 27 个特征。
        """
        history = np.asarray(history_values, dtype=np.float32)

        # 用这 12 个历史点做一个简单归一化
        mu = float(np.mean(history))
        sd = float(np.std(history))

        if sd < 1e-6:
            sd = 1.0

        z = (history - mu) / sd

        # lag_12 ... lag_1
        lags = z[-self.window_size:]

        # diff_11 ... diff_1
        diffs = np.diff(lags)

        roll_mean = np.mean(lags)
        roll_std = np.std(lags)

        current_step = self._get_current_step()
        tod = (current_step % STEPS_PER_DAY) / STEPS_PER_DAY

        sin_tod = np.sin(2 * np.pi * tod)
        cos_tod = np.cos(2 * np.pi * tod)

        features = np.hstack([
            lags,
            diffs,
            [roll_mean, roll_std, sin_tod, cos_tod]
        ]).astype(np.float32)

        # LightGBM 需要二维输入：[1, 27]
        return features.reshape(1, -1), mu, sd

    def predict(self, event: TrafficRecord) -> dict[str, Any]:
        start = time.perf_counter()

        sensor_id = event.sensor_id
        actual_flow = float(event.flow)

        history = self.sensor_memory[sensor_id]

        # 新模型需要至少 12 个历史点
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

        # 构造 27 个特征
        features, mu, sd = self._build_features(list(history))

        # 模型预测出来的是归一化后的 z 值
        predicted_z = float(self.model.predict(features)[0])

        # 转回原始 flow 数值
        predicted_flow = predicted_z * sd + mu

        flow_drop = predicted_flow - actual_flow

        # 预测完之后，再把当前真实值放进历史
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
