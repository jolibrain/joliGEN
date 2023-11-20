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
async def test_predict_endpoint_diffusion_success(dataroot, api):
    model_in_file = os.path.abspath(os.path.join(dataroot, "latest_net_G_A.pth"))
    dir_out = os.path.join(dataroot, "../")
    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "img_in": os.path.join(
                dataroot, "../horse2zebra/trainA/n02381460_1001.jpg"
            ),
            "dir_out": dir_out,
        }
    }

    response = api.post("/predict", json=payload)
    assert response.status_code == 200

    json_response = response.json()
    predict_name = json_response["name"]

    assert "message" in json_response
    assert "status" in json_response
    assert "name" in json_response
    assert json_response["message"] == "ok"
    assert json_response["status"] == "running"
    assert json_response["name"].startswith("predict_")
    assert len(json_response["name"]) > 0

    with api.websocket_connect(f"/ws/predict/%s" % predict_name) as ws:

        while True:

            try:

                data = ws.receive_json()

                if data["status"] != "log":
                    assert data["status"] == "stopped"
                    assert data["message"] == f"%s is stopped" % predict_name
                    break

                assert data["status"] == "log"
                assert "gen_single_image_diffusion" in data["message"]
                assert predict_name in data["message"]

                await asyncio.sleep(1)

            except WebSocketDisconnect:
                break

    for output in ["cond", "generated", "orig", "y_t"]:
        img_out = os.path.join(dir_out, f"%s_0_%s.png" % (predict_name, output))
        assert os.path.exists(img_out)
        if os.path.exists(img_out):
            os.remove(img_out)


def test_predict_endpoint_sync_success(dataroot, api):

    model_in_file = os.path.abspath(os.path.join(dataroot, "latest_net_G_A.pth"))
    dir_out = os.path.join(dataroot, "../")

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "img_in": os.path.join(
                dataroot, "../horse2zebra/trainA/n02381460_1001.jpg"
            ),
            "dir_out": dir_out,
        },
        "server": {"sync": True},
    }

    response = api.post("/predict", json=payload)
    assert response.status_code == 200

    json_response = response.json()
    predict_name = json_response["name"]

    assert "message" in json_response
    assert "status" in json_response
    assert "name" in json_response
    assert json_response["message"] == "ok"
    assert json_response["status"] == "stopped"
    assert json_response["name"].startswith("predict_")
    assert len(json_response["name"]) > 0

    for output in ["cond", "generated", "orig", "y_t"]:
        img_out = os.path.join(dir_out, f"%s_0_%s.png" % (predict_name, output))
        assert os.path.exists(img_out)
        if os.path.exists(img_out):
            os.remove(img_out)
