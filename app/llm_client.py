import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
LLM_OUTPUT_FILE = BASE_DIR / "data" / "llm_outputs.jsonl"

# 强制读取项目根目录下的 .env
load_dotenv(ENV_FILE, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GOOGLE_API_KEY and GEMINI_API_KEY:
    logger.info("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")

API_KEY = GOOGLE_API_KEY or GEMINI_API_KEY
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

LLM_TIMEOUT_SECONDS = 10.0
DEFAULT_COOLDOWN_SECONDS = 30.0

llm_lock = asyncio.Lock()
LLM_COOLDOWN_UNTIL = 0.0


def save_llm_output(record: Dict[str, Any]) -> None:
    """
    Save every LLM diagnosis result to data/llm_outputs.jsonl.

    JSONL format:
    - one JSON object per line
    - easy to append
    - easy to inspect later
    """
    try:
        LLM_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **record,
        }

        with open(LLM_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    except Exception as exc:
        logger.warning(f"Failed to save LLM output: {exc}")


def get_expert_fallback_diagnosis(sensor_id: str, severity: str, reason: str) -> str:
    """
    Local expert-rule fallback.

    This is used when:
    - Gemini API key is missing
    - Gemini is in cooldown
    - Gemini times out
    - Gemini API call fails
    """
    diagnoses = [
        (
            f"Fallback diagnosis: Detected {severity} flow abnormality on {sensor_id}. "
            f"Possible cause: road obstruction or temporary congestion. "
            f"Reason: {reason}. Recommended action: dispatch patrol or verify with nearby sensors."
        ),
        (
            f"Fallback diagnosis: Traffic anomaly detected on {sensor_id}. "
            f"Possible cause: congestion, sensor instability, or short-term traffic disruption. "
            f"Reason: {reason}. Recommended action: cross-check adjacent sensors."
        ),
        (
            f"Fallback diagnosis: Significant baseline deviation at {sensor_id}. "
            f"Possible cause: accident, road blockage, or abnormal demand change. "
            f"Reason: {reason}. Recommended action: initiate remote diagnostic check."
        ),
    ]

    return random.choice(diagnoses)


def build_prompt(
    event_data: Dict[str, Any],
    local_result: Optional[Dict[str, Any]] = None,
    decision_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a concise dynamic prompt for Gemini.

    The LLM is used for explanation / escalation support,
    not for every traffic prediction.
    """
    local_result = local_result or {}
    decision_result = decision_result or {}

    sensor_id = event_data.get("sensor_id", "Unknown")
    timestamp = event_data.get("timestamp", "Unknown")
    flow = event_data.get("flow", "Unknown")
    speed = event_data.get("speed", "Unknown")
    occupancy = event_data.get("occupancy", "Unknown")

    predicted_flow = local_result.get("predicted_flow", "Unknown")
    actual_flow = local_result.get("actual_flow", flow)
    anomaly_score = local_result.get("anomaly_score", "Unknown")
    local_status = local_result.get("status", "Unknown")

    severity = decision_result.get("severity_level", "MEDIUM")
    trigger_reason = decision_result.get("trigger_reason", "Unknown")

    return f"""
You are assisting a real-time traffic anomaly detection and AI benchmarking system.

Traffic event:
- sensor_id: {sensor_id}
- timestamp: {timestamp}
- flow: {flow}
- speed: {speed}
- occupancy: {occupancy}

Local model result:
- status: {local_status}
- actual_flow: {actual_flow}
- predicted_flow: {predicted_flow}
- anomaly_score: {anomaly_score}

Decision layer:
- severity_level: {severity}
- trigger_reason: {trigger_reason}

Please provide:
1. A concise explanation of why this event may be abnormal.
2. The likely operational meaning.
3. A recommended action for a traffic operator.

Keep the answer under 120 words.
""".strip()


async def call_gemini_async(prompt: str) -> str:
    """
    Call Gemini using the synchronous SDK inside a worker thread.

    This prevents blocking the FastAPI event loop.
    The import is inside the function so the whole app does not crash
    at startup if google-genai is not installed.
    """
    from google import genai

    def sync_call() -> str:
        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        text = getattr(response, "text", None)

        if not text:
            return "Gemini returned an empty response."

        return text.strip()

    return await asyncio.to_thread(sync_call)


async def diagnose_anomaly(
    event_data: Dict[str, Any],
    local_result: Dict[str, Any],
    decision_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Selective cloud LLM diagnosis.

    This function should only be called after the decision layer has decided
    that the event is anomalous, uncertain, high severity, or selected for benchmarking.
    """
    global LLM_COOLDOWN_UNTIL

    sensor_id = event_data.get("sensor_id", "Unknown")
    severity = decision_result.get("severity_level", "MEDIUM")
    trigger_reason = decision_result.get("trigger_reason", "Unknown")

    # 1. No API key: skip cloud call and use local expert fallback.
    if not API_KEY:
        explanation = get_expert_fallback_diagnosis(
            sensor_id=sensor_id,
            severity=severity,
            reason="Gemini API key is not configured",
        )

        result = {
            "llm_called": False,
            "llm_status": "skipped",
            "fallback_used": True,
            "llm_latency_ms": 0,
            "model": MODEL_NAME,
            "error": "GOOGLE_API_KEY or GEMINI_API_KEY is not configured",
            "explanation": explanation,
        }

        save_llm_output({
            "llm_status": "skipped",
            "fallback_used": True,
            "model": MODEL_NAME,
            "sensor_id": sensor_id,
            "severity": severity,
            "trigger_reason": trigger_reason,
            "event_data": event_data,
            "local_result": local_result,
            "decision_result": decision_result,
            "explanation": explanation,
            "error": "API key is not configured",
        })

        return result

    # 2. Cooldown check.
    now = time.time()

    if now < LLM_COOLDOWN_UNTIL:
        explanation = get_expert_fallback_diagnosis(
            sensor_id=sensor_id,
            severity=severity,
            reason="LLM cooldown is active",
        )

        result = {
            "llm_called": False,
            "llm_status": "cooldown",
            "fallback_used": True,
            "llm_latency_ms": 0,
            "model": MODEL_NAME,
            "cooldown_remaining_seconds": round(LLM_COOLDOWN_UNTIL - now, 3),
            "explanation": explanation,
        }

        save_llm_output({
            "llm_status": "cooldown",
            "fallback_used": True,
            "model": MODEL_NAME,
            "sensor_id": sensor_id,
            "severity": severity,
            "trigger_reason": trigger_reason,
            "cooldown_remaining_seconds": round(LLM_COOLDOWN_UNTIL - now, 3),
            "event_data": event_data,
            "local_result": local_result,
            "decision_result": decision_result,
            "explanation": explanation,
        })

        return result

    prompt = build_prompt(
        event_data=event_data,
        local_result=local_result,
        decision_result=decision_result,
    )

    # 3. Only one cloud LLM call at a time.
    async with llm_lock:
        # Re-check cooldown after acquiring the lock.
        now = time.time()

        if now < LLM_COOLDOWN_UNTIL:
            explanation = get_expert_fallback_diagnosis(
                sensor_id=sensor_id,
                severity=severity,
                reason="LLM cooldown is active",
            )

            result = {
                "llm_called": False,
                "llm_status": "cooldown",
                "fallback_used": True,
                "llm_latency_ms": 0,
                "model": MODEL_NAME,
                "cooldown_remaining_seconds": round(LLM_COOLDOWN_UNTIL - now, 3),
                "explanation": explanation,
            }

            save_llm_output({
                "llm_status": "cooldown",
                "fallback_used": True,
                "model": MODEL_NAME,
                "sensor_id": sensor_id,
                "severity": severity,
                "trigger_reason": trigger_reason,
                "cooldown_remaining_seconds": round(LLM_COOLDOWN_UNTIL - now, 3),
                "event_data": event_data,
                "local_result": local_result,
                "decision_result": decision_result,
                "explanation": explanation,
            })

            return result

        started = time.perf_counter()

        try:
            response_text = await asyncio.wait_for(
                call_gemini_async(prompt),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

            # Start cooldown after a successful real cloud call.
            LLM_COOLDOWN_UNTIL = time.time() + DEFAULT_COOLDOWN_SECONDS

            save_llm_output({
                "llm_status": "success",
                "fallback_used": False,
                "model": MODEL_NAME,
                "sensor_id": sensor_id,
                "severity": severity,
                "trigger_reason": trigger_reason,
                "llm_latency_ms": elapsed_ms,
                "event_data": event_data,
                "local_result": local_result,
                "decision_result": decision_result,
                "prompt": prompt,
                "explanation": response_text,
            })

            print()
            print("========== GEMINI OUTPUT SAVED ==========")
            print(f"Saved to: {LLM_OUTPUT_FILE}")
            print(response_text)
            print("=========================================")
            print()

            return {
                "llm_called": True,
                "llm_status": "success",
                "fallback_used": False,
                "llm_latency_ms": elapsed_ms,
                "model": MODEL_NAME,
                "explanation": response_text,
            }

        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

            logger.warning("Gemini API timeout. Switching to fallback diagnosis.")

            LLM_COOLDOWN_UNTIL = time.time() + DEFAULT_COOLDOWN_SECONDS

            explanation = get_expert_fallback_diagnosis(
                sensor_id=sensor_id,
                severity=severity,
                reason="Gemini request timed out",
            )

            save_llm_output({
                "llm_status": "timeout",
                "fallback_used": True,
                "model": MODEL_NAME,
                "sensor_id": sensor_id,
                "severity": severity,
                "trigger_reason": trigger_reason,
                "llm_latency_ms": elapsed_ms,
                "event_data": event_data,
                "local_result": local_result,
                "decision_result": decision_result,
                "prompt": prompt,
                "error": "Gemini request timed out",
                "explanation": explanation,
            })

            return {
                "llm_called": True,
                "llm_status": "timeout",
                "fallback_used": True,
                "llm_latency_ms": elapsed_ms,
                "model": MODEL_NAME,
                "error": "Gemini request timed out",
                "explanation": explanation,
            }

        except ImportError as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

            logger.warning("google-genai is not installed. Switching to fallback diagnosis.")

            explanation = get_expert_fallback_diagnosis(
                sensor_id=sensor_id,
                severity=severity,
                reason="google-genai dependency is missing",
            )

            save_llm_output({
                "llm_status": "skipped",
                "fallback_used": True,
                "model": MODEL_NAME,
                "sensor_id": sensor_id,
                "severity": severity,
                "trigger_reason": trigger_reason,
                "llm_latency_ms": elapsed_ms,
                "event_data": event_data,
                "local_result": local_result,
                "decision_result": decision_result,
                "prompt": prompt,
                "error": f"google-genai is not installed: {exc}",
                "explanation": explanation,
            })

            return {
                "llm_called": False,
                "llm_status": "skipped",
                "fallback_used": True,
                "llm_latency_ms": elapsed_ms,
                "model": MODEL_NAME,
                "error": f"google-genai is not installed: {exc}",
                "explanation": explanation,
            }

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

            logger.warning(f"Gemini API call failed. Switching to fallback diagnosis: {exc}")

            LLM_COOLDOWN_UNTIL = time.time() + DEFAULT_COOLDOWN_SECONDS

            explanation = get_expert_fallback_diagnosis(
                sensor_id=sensor_id,
                severity=severity,
                reason="Gemini API call failed",
            )

            save_llm_output({
                "llm_status": "error",
                "fallback_used": True,
                "model": MODEL_NAME,
                "sensor_id": sensor_id,
                "severity": severity,
                "trigger_reason": trigger_reason,
                "llm_latency_ms": elapsed_ms,
                "event_data": event_data,
                "local_result": local_result,
                "decision_result": decision_result,
                "prompt": prompt,
                "error": str(exc),
                "explanation": explanation,
            })

            return {
                "llm_called": True,
                "llm_status": "error",
                "fallback_used": True,
                "llm_latency_ms": elapsed_ms,
                "model": MODEL_NAME,
                "error": str(exc),
                "explanation": explanation,
            }
