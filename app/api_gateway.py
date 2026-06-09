from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.gateway import TrafficRecord
from app.dlq import write_dlq
from app.pipeline_service import process_valid_event


app = FastAPI(
    title="Real-Time Traffic Anomaly Detection Gateway",
    version="0.1.0",
    description="FastAPI ingestion gateway for validated traffic sensor events."
)


def sanitize_validation_errors(errors):
    """
    Convert Pydantic validation errors into JSON-safe format.

    Pydantic v2 may include ValueError objects inside ctx,
    which cannot be directly JSON serialized.
    """
    safe_errors = []

    for error in errors:
        safe_error = dict(error)

        if "ctx" in safe_error:
            safe_ctx = {}
            for key, value in safe_error["ctx"].items():
                safe_ctx[key] = str(value)
            safe_error["ctx"] = safe_ctx

        safe_errors.append(safe_error)

    return safe_errors


@app.get("/")
async def root():
    return {
        "message": "Traffic Anomaly Detection Gateway is running.",
        "docs": "/docs",
        "health": "/health",
        "ingestion_endpoint": "/v1/traffic/events"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "traffic-ingestion-gateway",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/traffic/events")
async def ingest_traffic_event(request: Request):
    """
    External traffic sensor events enter the system here.

    Flow:
    1. Receive raw JSON payload
    2. Validate with Pydantic
    3. Invalid payload -> DLQ
    4. Valid payload -> production pipeline
    """
    request_id = str(uuid4())
    received_at = datetime.now(timezone.utc).isoformat()

    try:
        raw_payload = await request.json()

    except Exception as exc:
        write_dlq(
            request_id=request_id,
            payload=None,
            error_type="malformed_json",
            error_reason=str(exc),
            source="fastapi_gateway"
        )

        return JSONResponse(
            status_code=400,
            content={
                "status": "rejected",
                "request_id": request_id,
                "reason": "malformed_json",
                "message": "Request body is not valid JSON."
            }
        )

    try:
        event = TrafficRecord(**raw_payload)

    except ValidationError as exc:
        safe_errors = sanitize_validation_errors(exc.errors())

        write_dlq(
            request_id=request_id,
            payload=raw_payload,
            error_type="validation_error",
            error_reason=safe_errors,
            source="fastapi_gateway"
        )

        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "request_id": request_id,
                "reason": "validation_error",
                "details": safe_errors
            }
        )

    pipeline_result = await process_valid_event(
        request_id=request_id,
        event=event,
        received_at=received_at
    )

    return {
        "status": "accepted",
        "request_id": request_id,
        "received_at": received_at,
        "pipeline_result": pipeline_result
    }