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


@pytest.mark.asyncio
async def test_predict_endpoint_gan_success(dataroot, api):

    model_in_file = os.path.abspath(os.path.join(dataroot, "latest_net_G_A.pth"))
    img_out = os.path.join(dataroot, "out_success.jpg")

    if os.path.exists(img_out):
        os.remove(img_out)

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "gan",
            "img_in": os.path.join(
                dataroot, "../horse2zebra/trainA/n02381460_1001.jpg"
            ),
            "img_out": img_out,
        }
    }

    response = api.post("/predict", json=payload)
    json_response = response.json()
    print(payload)
    print(response)
    print(json_response)
    assert response.status_code == 200

    assert "message" in json_response
    assert "status" in json_response
    assert "name" in json_response
    assert json_response["message"] == "ok"
    assert json_response["status"] == "running"
    assert json_response["name"].startswith("predict_")
    assert len(json_response["name"]) > 0

    predict_name = json_response["name"]
    with api.websocket_connect(f"/ws/predict/%s" % predict_name) as ws:

        while True:

            try:

                data = ws.receive_json()

                if data["status"] != "log":
                    assert data["status"] == "stopped"
                    assert data["message"] == f"%s is stopped" % predict_name
                    break

                assert data["status"] == "log"
                assert "gen_single_image" in data["message"]
                assert predict_name in data["message"]

                await asyncio.sleep(1)

            except WebSocketDisconnect:
                break

    assert os.path.exists(img_out)
    if os.path.exists(img_out):
        os.remove(img_out)


def test_predict_endpoint_sync_success(dataroot, api):

    model_in_file = os.path.abspath(os.path.join(dataroot, "latest_net_G_A.pth"))
    img_out = os.path.join(dataroot, "out_success_sync.jpg")

    if os.path.exists(img_out):
        os.remove(img_out)

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "gan",
            "img_in": os.path.join(
                dataroot, "../horse2zebra/trainA/n02381460_1001.jpg"
            ),
            "img_out": img_out,
        },
        "server": {"sync": True},
    }

    response = api.post("/predict", json=payload)
    assert response.status_code == 200

    json_response = response.json()
    assert "message" in json_response
    assert "status" in json_response
    assert "name" in json_response
    assert json_response["message"] == "ok"
    assert json_response["status"] == "stopped"
    assert json_response["name"].startswith("predict_")
    assert len(json_response["name"]) > 0

    assert os.path.exists(img_out)
    if os.path.exists(img_out):
        os.remove(img_out)
