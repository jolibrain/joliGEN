import asyncio
import pytest
import sys
import os
import shutil
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import base64

sys.path.append(sys.path[0] + "/..")
from server.joligen_api import app
import train
from options.train_options import TrainOptions


@pytest.fixture
def api():
    return TestClient(app)


@pytest.fixture(autouse=True)
def run_before_and_after_tests(dataroot):

    name = "joligen_utest_api_cut"
    print(dataroot)

    json_like_dict = {
        "name": name,
        "output_display_env": name,
        "dataroot": dataroot,
        "checkpoints_dir": "/".join(dataroot.split("/")[:-1]),
        "model_type": "cut",
        "G_netG": "mobile_resnet_attn",
        "output_print_freq": 1,
        "gpu_ids": "0",
        "G_lr": 0.0002,
        "D_lr": 0.0001,
        "data_crop_size": 64,
        "data_load_size": 64,
        "train_n_epochs": 1,
        "train_n_epochs_decay": 0,
        "data_dataset_mode": "unaligned_labeled_mask",
        "data_max_dataset_size": 10,
        "train_n_epochs": 1,
        "model_input_nc": 3,
        "model_output_nc": 3,
        "data_relative_paths": True,
    }
    opt = TrainOptions().parse_json(json_like_dict, save_config=True)
    train.launch_training(opt)

    yield

    try:
        model_dir = os.path.join(dataroot, "..", name)
        shutil.rmtree(model_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


@pytest.mark.asyncio
async def test_predict_endpoint_gan_success(dataroot, api):

    name = "joligen_utest_api_cut"
    dir_model = "/".join(dataroot.split("/")[:-1])

    if not os.path.exists(dir_model):
        pytest.fail("Model does not exist")

    model_in_file = os.path.abspath(os.path.join(dir_model, name, "latest_net_G_A.pth"))

    if not os.path.exists(model_in_file):
        pytest.fail(f"Model file does not exist: %s" % model_in_file)

    img_in = os.path.join(dataroot, "trainA", "img", "00000.png")

    if not os.path.exists(img_in):
        pytest.fail(f"Image input file does not exist: %s" % img_in)

    img_out = os.path.abspath(os.path.join(dir_model, "out_success_sync.jpg"))

    if os.path.exists(img_out):
        os.remove(img_out)

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "gan",
            "img_in": img_in,
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

    name = "joligen_utest_api_cut"
    dir_model = "/".join(dataroot.split("/")[:-1])

    if not os.path.exists(dir_model):
        pytest.fail("Model does not exist")

    model_in_file = os.path.abspath(os.path.join(dir_model, name, "latest_net_G_A.pth"))

    if not os.path.exists(model_in_file):
        pytest.fail(f"Model file does not exist: %s" % model_in_file)

    img_in = os.path.join(dataroot, "trainA", "img", "00000.png")

    if not os.path.exists(img_in):
        pytest.fail(f"Image input file does not exist: %s" % img_in)

    img_out = os.path.abspath(os.path.join(dir_model, "out_success_sync.jpg"))

    if os.path.exists(img_out):
        os.remove(img_out)

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "gan",
            "img_in": img_in,
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


def test_predict_endpoint_sync_base64(dataroot, api):

    name = "joligen_utest_api_cut"
    dir_model = "/".join(dataroot.split("/")[:-1])

    if not os.path.exists(dir_model):
        pytest.fail("Model does not exist")

    model_in_file = os.path.abspath(os.path.join(dir_model, name, "latest_net_G_A.pth"))

    if not os.path.exists(model_in_file):
        pytest.fail(f"Model file does not exist: %s" % model_in_file)

    img_in = os.path.join(dataroot, "trainA", "img", "00000.png")

    if not os.path.exists(img_in):
        pytest.fail(f"Image input file does not exist: %s" % img_in)

    img_out = os.path.abspath(os.path.join(dir_model, "out_success_sync.jpg"))

    if os.path.exists(img_out):
        os.remove(img_out)

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "img_in": img_in,
            "img_out": img_out,
        },
        "server": {"sync": True, "base64": True},
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

    assert len(json_response["base64"]) == 1
    with open(img_out, "rb") as f:
        base64_out = base64.b64encode(f.read()).decode("utf-8")
        assert base64_out == json_response["base64"][0]

    if os.path.exists(img_out):
        os.remove(img_out)
