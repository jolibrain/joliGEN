import asyncio
import pytest
import sys
import os
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.append(sys.path[0] + "/..")
from server.joligen_api import app


@pytest.fixture
def api():
    return TestClient(app)


def test_predict_endpoint_missing_options(api):
    response = api.post("/predict", json={})
    assert response.status_code == 400
    assert response.json() == {
        "detail": "parameter predict_options is required",
    }

    response = api.post("/predict", json={"predict_options": {}})
    assert response.status_code == 400
    assert response.json() == {
        "detail": "parameter predict_options.model_in_file is required",
    }

    response = api.post(
        "/predict", json={"predict_options": {"model_in_file": "unknown"}}
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "parameter predict_options.img_in is required",
    }

    payload = {"predict_options": {"model_in_file": "unknown", "img_in": "unknown"}}
    response = api.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json() == {
        "detail": "train_config.json not found",
    }

    with api.websocket_connect("/ws/predict/random") as ws:
        data = ws.receive_json()
        assert data == {"status": "error", "message": "random not in context"}
