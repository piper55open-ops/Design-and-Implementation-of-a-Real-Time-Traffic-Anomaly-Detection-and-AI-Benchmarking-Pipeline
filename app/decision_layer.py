import random
from typing import Any, Dict, Optional


class DecisionLayer:
    """
    Formal local-first decision layer.

    This layer decides whether a traffic event should:
    1. Stay local-only
    2. Be escalated to the cloud LLM

    Important:
    LightGBM regression does not provide true confidence.
    Therefore, confidence_proxy is an engineering approximation based on prediction error ratio.
    """

    def __init__(
        self,
        high_drop_threshold: float = 12.0,
        medium_drop_threshold: float = 6.0,
        high_drop_ratio: float = 0.40,
        low_confidence_threshold: float = 0.50,
        benchmark_sample_rate: float = 0.00,

    ):
        self.high_drop_threshold = high_drop_threshold
        self.medium_drop_threshold = medium_drop_threshold
        self.high_drop_ratio = high_drop_ratio
        self.low_confidence_threshold = low_confidence_threshold
        self.benchmark_sample_rate = benchmark_sample_rate

    def decide(self, local_result: Dict[str, Any]) -> Dict[str, Any]:
        model_status = local_result.get("model_status")

        if model_status != "predicted":
            return {
                "decision": "LOCAL_ONLY",
                "escalate_to_llm": False,
                "trigger_reason": "MODEL_WARMUP",
                "severity_level": "NONE",
                "severity_score": 0.0,
                "confidence_proxy": None,
                "benchmark_sample": False,
                "explanation": "Local model is still warming up, so no LLM escalation is triggered."
            }

        actual_flow = local_result.get("actual_flow")
        predicted_flow = local_result.get("predicted_flow")
        flow_drop = local_result.get("flow_drop")

        if actual_flow is None or predicted_flow is None or flow_drop is None:
            return {
                "decision": "LOCAL_ONLY",
                "escalate_to_llm": False,
                "trigger_reason": "MISSING_MODEL_OUTPUT",
                "severity_level": "UNKNOWN",
                "severity_score": 0.0,
                "confidence_proxy": None,
                "benchmark_sample": False,
                "explanation": "Model output is incomplete, so the event remains local-only."
            }

        predicted_flow = float(predicted_flow)
        actual_flow = float(actual_flow)
        flow_drop = float(flow_drop)

        # Positive flow_drop means actual traffic flow is lower than predicted.
        # This may indicate congestion, incident, or sensor abnormality.
        drop_ratio = flow_drop / max(predicted_flow, 1.0)

        # Only positive drop is treated as traffic flow drop severity.
        severity_score = max(0.0, drop_ratio)

        if flow_drop >= self.high_drop_threshold or drop_ratio >= self.high_drop_ratio:
            severity_level = "HIGH"
        elif flow_drop >= self.medium_drop_threshold:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        # This is not real statistical confidence.
        # It is a proxy based on how far actual flow is from predicted flow.
        error_ratio = abs(predicted_flow - actual_flow) / max(predicted_flow, 1.0)
        confidence_proxy = max(0.0, round(1.0 - min(error_ratio, 1.0), 3))

        benchmark_sample = random.random() < self.benchmark_sample_rate

        if severity_level == "HIGH":
            return {
                "decision": "ESCALATE_TO_LLM",
                "escalate_to_llm": True,
                "trigger_reason": "HIGH_SEVERITY_FLOW_DROP",
                "severity_level": severity_level,
                "severity_score": round(severity_score, 3),
                "confidence_proxy": confidence_proxy,
                "benchmark_sample": benchmark_sample,
                "explanation": "Actual flow is significantly lower than predicted flow."
            }

        if confidence_proxy < self.low_confidence_threshold:
            return {
                "decision": "ESCALATE_TO_LLM",
                "escalate_to_llm": True,
                "trigger_reason": "LOW_CONFIDENCE_PROXY",
                "severity_level": severity_level,
                "severity_score": round(severity_score, 3),
                "confidence_proxy": confidence_proxy,
                "benchmark_sample": benchmark_sample,
                "explanation": "Prediction error is large, so this event is treated as uncertain."
            }

        if benchmark_sample:
            return {
                "decision": "ESCALATE_TO_LLM",
                "escalate_to_llm": True,
                "trigger_reason": "BENCHMARK_SAMPLE",
                "severity_level": severity_level,
                "severity_score": round(severity_score, 3),
                "confidence_proxy": confidence_proxy,
                "benchmark_sample": benchmark_sample,
                "explanation": "This event is selected for benchmark comparison."
            }

        return {
            "decision": "LOCAL_ONLY",
            "escalate_to_llm": False,
            "trigger_reason": "NONE",
            "severity_level": severity_level,
            "severity_score": round(severity_score, 3),
            "confidence_proxy": confidence_proxy,
            "benchmark_sample": benchmark_sample,
            "explanation": "Local model result is sufficient; no cloud LLM call is needed."
        }


decision_layer = DecisionLayer()