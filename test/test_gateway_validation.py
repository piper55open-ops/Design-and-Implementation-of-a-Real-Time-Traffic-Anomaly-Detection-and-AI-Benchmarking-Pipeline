import pytest
from pydantic import ValidationError

from app.gateway import TrafficRecord


def test_valid_payload_should_pass():
    record = TrafficRecord(
        sensor_id="S-001",
        timestamp="08:00:00",
        flow=45.0,
        speed=60.0,
        occupancy=20.0,
    )

    assert record.sensor_id == "S-001"
    assert record.timestamp == "08:00:00"
    assert record.flow == 45.0


def test_invalid_timestamp_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="not-a-time",
            flow=45.0,
        )


def test_numeric_string_flow_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="08:00:00",
            flow="45",
        )


def test_extra_field_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="08:00:00",
            flow=45.0,
            evil="extra-field",
        )


def test_negative_flow_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="08:00:00",
            flow=-10,
        )


def test_occupancy_out_of_range_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="08:00:00",
            flow=45.0,
            occupancy=150,
        )


def test_invalid_latitude_should_fail():
    with pytest.raises(ValidationError):
        TrafficRecord(
            sensor_id="S-001",
            timestamp="08:00:00",
            flow=45.0,
            latitude=200,
        )
