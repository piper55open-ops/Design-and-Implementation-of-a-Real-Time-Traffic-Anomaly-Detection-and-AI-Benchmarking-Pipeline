from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional

class TrafficRecord(BaseModel):
    sensor_id: str = Field(..., description="Sensor unique identifier")
    timestamp: str = Field(..., description="Data collection timestamp")
    flow: float = Field(..., description="Traffic flow (vehicles/hour)")
    speed: Optional[float] = Field(None, description="Average speed (km/h)")
    occupancy: Optional[float] = Field(None, description="Lane occupancy rate (%)")

    @field_validator('speed')
    def validate_speed(cls, value):
        if value is not None:
            if value < 0:
                raise ValueError(f"Negative speed detected: {value}")
            if value > 250:
                raise ValueError(f"Speed exceeds physical limit: {value}")
        return value

    @field_validator('occupancy')
    def validate_occupancy(cls, value):
        if value is not None:
            if value < 0 or value > 100:
                raise ValueError(f"Occupancy out of bounds (0-100): {value}")
        return value