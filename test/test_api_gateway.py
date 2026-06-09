from fastapi.testclient import TestClient

from app.api_gateway import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200


def test_valid_payload_should_be_accepted():
    payload = {
        "sensor_id": "S-001",
        "timestamp": "08:00:00",
        "flow": 45.0,
        "speed": 60.0,
        "occupancy": 20.0,
    }

    response = client.post("/v1/traffic/events", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert "request_id" in body


def test_bad_timestamp_should_be_rejected():
    payload = {
        "sensor_id": "S-001",
        "timestamp": "not-a-time",
        "flow": 45.0,
    }

    response = client.post("/v1/traffic/events", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "validation_error"


def test_numeric_string_flow_should_be_rejected():
    payload = {
        "sensor_id": "S-001",
        "timestamp": "08:00:00",
        "flow": "45",
    }

    response = client.post("/v1/traffic/events", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "validation_error"


def test_extra_field_should_be_rejected():
    payload = {
        "sensor_id": "S-001",
        "timestamp": "08:00:00",
        "flow": 45.0,
        "evil": "extra-field",
    }

    response = client.post("/v1/traffic/events", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "validation_error"


def test_broken_json_should_be_rejected():
    response = client.post(
        "/v1/traffic/events",
        content='{"sensor_id": "S-001", "timestamp": "08:00:00", "flow": ',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "malformed_json"
