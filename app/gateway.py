from datetime import datetime, time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_REASONABLE_FLOW = 100000.0
MAX_REASONABLE_SPEED_KMH = 200.0


class TrafficRecord(BaseModel):
    """
    Strict traffic sensor payload schema.

    This validation gateway is the first defensive layer before any AI inference.
    Invalid payloads should be rejected by FastAPI and routed to the DLQ by api_gateway.py.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    sensor_id: str = Field(
        ...,
        min_length=1,
        description="Traffic sensor identifier. Must not be empty.",
    )

    timestamp: str = Field(
        ...,
        min_length=1,
        description="Timestamp in HH:MM:SS or ISO datetime format.",
    )

    flow: float = Field(
        ...,
        ge=0,
        le=MAX_REASONABLE_FLOW,
        description="Traffic flow value. Must be a non-negative numeric value.",
    )

    speed: Optional[float] = Field(
        default=None,
        ge=0,
        le=MAX_REASONABLE_SPEED_KMH,
        description="Vehicle speed in km/h. Must be non-negative if provided.",
    )

    occupancy: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Road occupancy percentage. Must be between 0 and 100 if provided.",
    )

    # Optional fields are included to match the proposal,
    # but they are not required because the current PEMS03 simulation may not send them.
    road_segment: Optional[str] = Field(
        default=None,
        description="Optional road segment identifier.",
    )

    latitude: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Optional latitude value.",
    )

    longitude: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180,
        description="Optional longitude value.",
    )

    @field_validator("sensor_id", "timestamp", "road_segment", mode="before")
    @classmethod
    def reject_non_string_values(cls, value, info):
        """
        Reject non-string values for text fields.
        This prevents values such as sensor_id=123 from being silently coerced to "123".
        """
        if value is None:
            return value

        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")

        if value.strip() == "":
            raise ValueError(f"{info.field_name} must not be empty")

        return value.strip()

    @field_validator(
        "flow",
        "speed",
        "occupancy",
        "latitude",
        "longitude",
        mode="before",
    )
    @classmethod
    def reject_string_or_boolean_numbers(cls, value, info):
        """
        Reject numeric strings and booleans.

        Examples rejected:
        flow="50"
        speed="60.5"
        occupancy=True

        JSON numeric values such as 50 and 50.5 are accepted.
        """
        if value is None:
            return value

        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be numeric, not boolean")

        if isinstance(value, str):
            raise ValueError(f"{info.field_name} must be numeric, not string")

        if not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a numeric value")

        return value

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, value: str):
        """
        Accept either:
        - HH:MM:SS format, e.g. "08:15:30"
        - ISO datetime format, e.g. "2026-05-16T08:15:30"

        Reject values such as:
        - "not-a-time"
        - "next Monday"
        - "25:99:00"
        """
        parsed = False

        try:
            time.fromisoformat(value)
            parsed = True
        except ValueError:
            pass

        if not parsed:
            try:
                datetime.fromisoformat(value)
                parsed = True
            except ValueError:
                pass

        if not parsed:
            raise ValueError(
                "timestamp must be HH:MM:SS or ISO datetime format"
            )

        return value
