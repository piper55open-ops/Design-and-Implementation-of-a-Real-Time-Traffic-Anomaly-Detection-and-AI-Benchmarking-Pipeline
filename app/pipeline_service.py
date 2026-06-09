import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from app.llm_client import diagnose_anomaly, LLM_COOLDOWN_UNTIL # 确保能引用这个变量
import time
from app.gateway import TrafficRecord
from app.event_log import write_benchmark_event
from app.model_service import local_model_service
from app.decision_layer import decision_layer
from app.llm_client import diagnose_anomaly

logger = logging.getLogger(__name__)

LIVE_DATA_FILE = Path("data/live_traffic.json")

# ==========================================
# 💡 终极杀招：内存级状态锁 (彻底杜绝高并发文件覆盖)
GLOBAL_DASHBOARD_STATE = {
    "history": [],
    "current_alert": None,
    "llm_report": None
}


# ==========================================

def update_dashboard_state(event_data: dict, local_result: dict, decision_result: dict, llm_result: dict = None):
    sensor = event_data.get("sensor_id")

    # 1. 直接更新内存里的历史流量
    GLOBAL_DASHBOARD_STATE["history"].append({
        "timestamp": event_data.get("timestamp"),
        "flow": event_data.get("flow"),
        "sensor_id": sensor
    })
    GLOBAL_DASHBOARD_STATE["history"] = GLOBAL_DASHBOARD_STATE["history"][-100:]

    # 2. 更新内存里的警报状态
    if decision_result.get("escalate_to_llm"):
        GLOBAL_DASHBOARD_STATE["current_alert"] = {
            "sensor": sensor,
            "expected": local_result.get("predicted_flow", 0.0),
            "actual": event_data.get("flow")
        }
    else:
        GLOBAL_DASHBOARD_STATE["current_alert"] = None
        GLOBAL_DASHBOARD_STATE["llm_report"] = None

    # 3. 更新内存里的大模型报告
    if llm_result and llm_result.get("llm_status") == "success":
        GLOBAL_DASHBOARD_STATE["llm_report"] = llm_result.get("explanation")
    elif llm_result and llm_result.get("llm_status") in ["error", "quota_exceeded", "cooldown"]:
        # 写入友好的降级提示
        GLOBAL_DASHBOARD_STATE[
            "llm_report"] = f"⚠️ 云端服务繁忙，系统已自动降级为本地预警模式。 (详情: {llm_result.get('explanation', '限流')})"

    # 4. 单向输出到硬盘，只给 Streamlit 大屏看（绝对不会再被互相覆盖）
    try:
        with open(LIVE_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(GLOBAL_DASHBOARD_STATE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"写入大屏文件失败: {e}")


async def run_llm_escalation_job(
        request_id: str,
        event_data: dict[str, Any],
        local_result: dict[str, Any],
        decision_result: dict[str, Any]
) -> None:
    started_at = datetime.now(timezone.utc).isoformat()

    llm_result = await diagnose_anomaly(
        event_data=event_data,
        local_result=local_result,
        decision_result=decision_result
    )

    completed_at = datetime.now(timezone.utc).isoformat()

    write_benchmark_event({
        "event_type": "llm_escalation_result",
        "request_id": request_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "pipeline_stage": "background_llm_escalation",
        "event": event_data,
        "local_model": local_result,
        "decision": decision_result,
        "llm": llm_result
    })

    update_dashboard_state(event_data, local_result, decision_result, llm_result)


async def process_valid_event(
        request_id: str,
        event: TrafficRecord,
        received_at: str
) -> dict[str, Any]:
    event_data = event.model_dump()

    local_result = local_model_service.predict(event)
    decision_result = decision_layer.decide(local_result)

    llm_result = {
        "llm_called": False,
        "llm_status": "skipped",
        "llm_latency_ms": 0,
        "trigger_reason": decision_result.get("trigger_reason"),
        "explanation": "LLM was not triggered because the event was handled locally."
    }

    # Track A 先行更新大屏
    update_dashboard_state(event_data, local_result, decision_result)

    if decision_result.get("escalate_to_llm") is True:
        llm_result = {
            "llm_called": True,
            "llm_status": "pending",
            "llm_latency_ms": None,
            "trigger_reason": decision_result.get("trigger_reason"),
            "explanation": "LLM escalation has been scheduled as a background task."
        }
        asyncio.create_task(
            run_llm_escalation_job(
                request_id=request_id,
                event_data=event_data,
                local_result=local_result,
                decision_result=decision_result
            )
        )

    result = {
        "event_type": "local_pipeline_result",
        "request_id": request_id,
        "received_at": received_at,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status": "processed",
        "pipeline_stage": "local_first_selective_llm",
        "event": event_data,
        "local_model": local_result,
        "decision": decision_result,
        "llm": llm_result
    }

    write_benchmark_event(result)
    return result