from fastapi import Request, FastAPI, HTTPException, WebSocket
import logging
import asyncio
import traceback
import json
import subprocess
import os
import shutil
import time
from pathlib import Path

import torch.multiprocessing as mp

from train import launch_training
from options.train_options import TrainOptions
from data import create_dataset
from enum import Enum
from pydantic import create_model, BaseModel, Field

from options.predict_options import PredictOptions

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from gen_single_image import launch_predict_single_image
from gen_single_image_diffusion import launch_predict_diffusion

from multiprocessing import Process

git_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__))
    .decode("ascii")
    .strip()
)
print("Launching JoliGEN Server\ncommit=%s" % git_hash, flush=True)

version_str = "*commit:* [%s](https://github.com/jolibrain/joliGEN/commit/%s)\n\n" % (
    git_hash[:8],
    git_hash,
)
description = (
    version_str
    + """This is the JoliGEN server API documentation.
"""
)
app = FastAPI(title="JoliGEN server", description=description)


# Additional schema

## TrainOptions
class ServerTrainOptions(BaseModel):
    sync: bool = Field(
        False,
        description="if false, the call returns immediately and train process "
        "is executed in the background. If true, the call returns only "
        "when training process is finished",
    )


class TrainBody(BaseModel):
    server: ServerTrainOptions = ServerTrainOptions()


TrainBodySchema = TrainBody.schema()
TrainBodySchema["properties"]["train_options"] = TrainOptions().get_schema()

# Ensure schema is valid at startup
json.dumps(
    TrainBodySchema,
    ensure_ascii=False,
    indent=None,
    separators=(",", ":"),
).encode("utf-8")

## PredictOptions
class ServerPredictOptions(BaseModel):
    sync: bool = Field(
        False,
        description="if false, the call returns immediately and inference process "
        "is executed in the background. If true, the call returns only "
        "when inference process is finished",
    )


class PredictBody(BaseModel):
    server: ServerPredictOptions = ServerPredictOptions()


PredictBodySchema = PredictBody.schema()
PredictBodySchema["properties"]["predict_options"] = PredictOptions().get_schema()

# Ensure schema is valid at startup
json.dumps(
    PredictBodySchema,
    ensure_ascii=False,
    indent=None,
    separators=(",", ":"),
).encode("utf-8")


generic_openapi = app.openapi


def custom_openapi():
    if not app.openapi_schema:
        app.openapi_schema = generic_openapi()
        app.openapi_schema["components"]["schemas"]["TrainOptions"] = TrainBodySchema
        app.openapi_schema["definitions"] = {}
        app.openapi_schema["definitions"][
            "ServerTrainOptions"
        ] = ServerTrainOptions.schema()
    return app.openapi_schema


app.openapi = custom_openapi


# Context variables
ctx = {}

LOG_PATH = os.environ.get(
    "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


async def log_reader_last_line(job_name):
    global LOG_PATH

    log_file = Path(f"{LOG_PATH}/{job_name}.log")

    if not log_file.exists() or log_file.stat().st_size == 0:
        return ""

    with open(log_file, "r") as f:
        try:
            return f.readlines()[-1]
        except Exception as e:
            print(f"error reading logs for {job_name}: {e}")
            return ""


@app.websocket("/ws/logs/{job_name}")
async def websocket_endpoint_log(job_name: str, websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(1)
            logs = await log_reader_last_line(job_name)

            # close ws if last log line contains success message
            if "success" in logs:
                print("success in logs, closing websocket")
                await websocket.send_text(logs)
                await websocket.close()
                break
            else:
                await websocket.send_text(logs)
    except Exception as e:
        print(f"error on ws log endpoint for {job_name}: {e}")
        print(e)


def stop_training(process):
    process.terminate()

    try:
        process.join()
    except Exception as e:
        print(e)


def is_alive(process):
    return process.is_alive()


@app.post(
    "/predict",
    status_code=201,
    summary="Start an inference process.",
    description="The inference process will be created using the same options as command line",
)
async def predict(request: Request):
    predict_body = await request.json()

    parser = PredictOptions()
    try:
        predict_method = predict_body["predict_options"]["predict-method"]
        opt = parser.parse_json(predict_body["predict_options"], save_config=False)

        # Parse the remaining options
        del predict_body["predict_options"]
        predict_body = PredictBody.parse_obj(predict_body)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="{0}".format(e))

    target = None
    if predict_method == "gen_single_image":
        target = launch_predict_single_image
    elif predict_method == "gen_single_image_diffusion":
        target = launch_predict_diffusion

    name = "predict_{}".format(int(time.time()))

    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    Path(f"{LOG_PATH}/{name}.log").touch()

    ctx[name] = Process(target=target, args=(opt, name))
    ctx[name].start()

    if predict_body.server.sync:
        try:
            # XXX could be awaited
            ctx[name].join()
        except Exception as e:
            return {"predict_name": name, "message": str(e), "status": "error"}
        del ctx[name]
        return {"message": "ok", "predict_name": name, "status": "stopped"}

    return {"message": "ok", "predict_name": name, "status": "running"}


@app.post(
    "/train/{name}",
    status_code=201,
    summary="Start a training process with given name.",
    description="The training process will be created using the same options as command line",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TrainOptions"}
                }
            }
        }
    },
)
async def train(name: str, request: Request):
    train_body = await request.json()

    parser = TrainOptions()
    try:
        opt = parser.parse_json(train_body["train_options"], save_config=True)

        # Parse the remaining options
        del train_body["train_options"]
        train_body = TrainBody.parse_obj(train_body)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="{0}".format(e))

    ctx[name] = Process(target=launch_training, args=(opt,))
    ctx[name].start()

    if train_body.server.sync:
        try:
            # XXX could be awaited
            ctx[name].join()
        except Exception as e:
            return {"name": name, "message": str(e), "status": "error"}
        del ctx[name]
        return {"message": "ok", "name": name, "status": "stopped"}

    return {"message": "ok", "name": name, "status": "running"}


@app.get(
    "/train/{name}", status_code=200, summary="Get the status of a training process"
)
async def get_train(name: str):
    if name in ctx:
        status = "running" if is_alive(ctx[name]) else "stopped"
        return {"status": status, "name": name}
    else:
        raise HTTPException(status_code=404, detail="Not found")


@app.get("/train", status_code=200, summary="Get the status of all training processes")
async def get_train_processes():
    processes = []
    for name in ctx:
        status = "running" if is_alive(ctx[name]) else "stopped"
        processes.append({"status": status, "name": name})

    return {"processes": processes}


@app.get("/info", status_code=200, summary="Get the server status")
async def get_info():
    return {"JoliGEN": version_str}


@app.delete(
    "/train/{name}",
    status_code=200,
    summary="Delete a training process.",
    description="If the process is running, it will be stopped.",
)
async def delete_train(name: str):
    if name in ctx:
        stop_training(ctx[name])
        del ctx[name]
        return {"message": "ok", "name": name}
    else:
        raise HTTPException(status_code=404, detail="Not found")


@app.delete(
    "/fs/",
    status_code=200,
    summary="Delete a file or a directory in the filesystem",
    description="This endpoint can be dangerous, use it with extreme caution",
)
async def delete_path(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return {"message": "ok"}
