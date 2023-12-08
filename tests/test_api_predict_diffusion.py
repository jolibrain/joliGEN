import asyncio
import pytest
import sys
import os
import shutil
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from PIL import Image

sys.path.append(sys.path[0] + "/..")
from server.joligen_api import app
import train
from options.train_options import TrainOptions


@pytest.fixture
def api():
    return TestClient(app)


@pytest.fixture(autouse=True)
def run_before_and_after_tests(dataroot):

    name = "joligen_utest_api_palette"

    json_like_dict = {
        "name": name,
        "dataroot": dataroot,
        "checkpoints_dir": "/".join(dataroot.split("/")[:-1]),
        "model_type": "palette",
        "output_display_env": name,
        "output_display_id": 0,
        "gpu_ids": "0",
        "data_dataset_mode": "self_supervised_labeled_mask",
        "data_load_size": 128,
        "data_crop_size": 128,
        "train_n_epochs": 1,
        "train_n_epochs_decay": 0,
        "data_max_dataset_size": 10,
        "data_relative_paths": True,
        "train_G_ema": True,
        "dataaug_no_rotate": True,
        "G_unet_mha_num_head_channels": 16,
        "G_unet_mha_channel_mults": [1, 2],
        "G_nblocks": 1,
        "G_padding_type": "reflect",
        "G_netG": "uvit",
        "G_unet_mha_norm_layer": "batchnorm",
        "G_unet_mha_vit_efficient": True,
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
async def test_predict_endpoint_diffusion_success(dataroot, api):

    name = "joligen_utest_api_palette"
    dir_model = "/".join(dataroot.split("/")[:-1])

    if not os.path.exists(dir_model):
        pytest.fail("Model does not exist")

    model_in_file = os.path.abspath(os.path.join(dir_model, name, "latest_net_G_A.pth"))

    if not os.path.exists(model_in_file):
        pytest.fail(f"Model file does not exist: %s" % model_in_file)

    img_in = os.path.join(dataroot, "trainA", "img", "00000.png")

    if not os.path.exists(img_in):
        pytest.fail(f"Image input file does not exist: %s" % img_in)

    img_resized = os.path.join(dataroot, "img_resized.jpg")
    img_to_resize = Image.open(img_in)
    img_to_resize.thumbnail((128, 128), Image.Resampling.LANCZOS)
    img_to_resize.save(img_resized, "JPEG")

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "palette",
            "img_in": img_resized,
            "dir_out": dir_model,
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
        img_out = os.path.abspath(
            os.path.join(dir_model, f"%s_0_%s.png" % (predict_name, output))
        )
        assert os.path.exists(img_out)
        if os.path.exists(img_out):
            os.remove(img_out)

    os.remove(img_resized)


def test_predict_endpoint_sync_success(dataroot, api):

    name = "joligen_utest_api_palette"
    dir_model = "/".join(dataroot.split("/")[:-1])

    if not os.path.exists(dir_model):
        pytest.fail("Model does not exist")

    model_in_file = os.path.abspath(os.path.join(dir_model, name, "latest_net_G_A.pth"))

    if not os.path.exists(model_in_file):
        pytest.fail(f"Model file does not exist: %s" % model_in_file)

    img_in = os.path.join(dataroot, "trainA", "img", "00000.png")

    if not os.path.exists(img_in):
        pytest.fail(f"Image input file does not exist: %s" % img_in)

    img_resized = os.path.join(dataroot, "img_resized.jpg")
    img_to_resize = Image.open(img_in)
    img_to_resize.thumbnail((128, 128), Image.Resampling.LANCZOS)
    img_to_resize.save(img_resized, "JPEG")

    payload = {
        "predict_options": {
            "model_in_file": model_in_file,
            "model_type": "palette",
            "img_in": img_resized,
            "dir_out": dir_model,
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
        img_out = os.path.abspath(
            os.path.join(dir_model, f"%s_0_%s.png" % (predict_name, output))
        )
        assert os.path.exists(img_out)
        if os.path.exists(img_out):
            os.remove(img_out)

    os.remove(img_resized)
