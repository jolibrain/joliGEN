from fastapi import Request, FastAPI, HTTPException, WebSocket
import asyncio
import traceback
import json
import subprocess
import os
import shutil
from pathlib import Path
import time

import torch.multiprocessing as mp

mp.set_start_method("spawn")

from train import launch_training
from options.train_options import TrainOptions
from data import create_dataset
from enum import Enum
from pydantic import create_model, BaseModel, Field

from multiprocessing import Process

from options.inference_gan_options import InferenceGANOptions
from options.inference_diffusion_options import InferenceDiffusionOptions

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from gen_single_image import inference as gan_inference
from gen_single_image_diffusion import inference as diffusion_inference

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


def stop_training(process):
    process.terminate()

    try:
        process.join()
    except Exception as e:
        print(e)


def is_alive(process):
    return process.is_alive()


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


# Inference

LOG_PATH = os.environ.get(
    "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


async def log_reader_last_line(name):
    global LOG_PATH

    log_file = Path(f"{LOG_PATH}/{name}.log")

    if not log_file.exists() or log_file.stat().st_size == 0:
        return ""

    with open(log_file, "r") as f:
        try:
            return f.readlines()[-1]
        except Exception as e:
            raise e


@app.websocket("/ws/predict/{name}")
async def websocket_predict_endpoint(ws: WebSocket, name: str):
    await ws.accept()

    try:
        while True:

            # error handling on name parameter
            if name not in ctx:

                await ws.send_json(
                    {"status": "error", "message": f"%s not in context" % name}
                )
                await ws.close()
                break

            elif not is_alive(ctx[name]):
                await ws.send_json(
                    {"status": "stopped", "message": f"%s is stopped" % name}
                )
                await ws.close()
                break

            # read last line of named inference log file
            try:
                log_line = await log_reader_last_line(name)
            except Exception as e:
                await ws.send_json(
                    {"status": "error", "message": f"log reading error on {name}: {e}"}
                )
                await ws.close()
                break

            # send last line to client
            if log_line != "":
                await ws.send_json({"status": "log", "message": log_line.strip()})

            # close connection if inference is finished
            if "success" in log_line or "error" in log_line:
                await ws.close()
                break

            # wait 1 second before next iteration
            await asyncio.sleep(1)

    except Exception as e:
        print(f"error on ws log endpoint for {name}: {e}")
        traceback.print_exc()
        await ws.send_json(
            {"status": "error", "message": f"error on ws log endpoint for {name}: {e}"}
        )
        await ws.close()


@app.post(
    "/predict",
    status_code=200,
    summary="Start a inference process",
    description="The inference process will be created using the same options as command line",
)
async def predict(request: Request):
    predict_body = await request.json()

    if "predict_options" not in predict_body:
        raise HTTPException(
            status_code=400, detail="parameter predict_options is required"
        )

    if "model_in_file" not in predict_body["predict_options"]:
        raise HTTPException(
            status_code=400,
            detail="parameter predict_options.model_in_file is required",
        )

    if "img_in" not in predict_body["predict_options"]:
        raise HTTPException(
            status_code=400, detail="parameter predict_options.img_in is required"
        )

    train_json_path = Path(
        os.path.dirname(predict_body["predict_options"]["model_in_file"]),
        "train_config.json",
    )

    if not train_json_path.exists():
        raise HTTPException(status_code=400, detail="train_config.json not found")

    try:
        with open(train_json_path, "r") as jsonf:
            train_json = json.load(jsonf)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="{0}".format(e))

    if "model_type" in train_json and train_json["model_type"] == "palette":
        target = diffusion_inference
        parser = InferenceDiffusionOptions()
    else:
        target = gan_inference
        parser = InferenceGANOptions()

    try:
        opt = parser.parse_json(predict_body["predict_options"], save_config=False)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="{0}".format(e))

    opt.name = "predict_{}".format(int(time.time()))

    ctx[opt.name] = mp.Process(target=target, args=(opt,))
    try:
        ctx[opt.name].start()
    except Exception as e:
        raise HTTPException(status_code=400, detail="{0}".format(e))

    if (
        "server" in predict_body
        and "sync" in predict_body["server"]
        and predict_body["server"]["sync"]
    ):

        # run in synchronous mode
        try:
            ctx[opt.name].join()
            return {"message": "ok", "name": opt.name, "status": "stopped"}
        except Exception as e:
            raise HTTPException(status_code=400, detail="{0}".format(e))

    else:

        # run in async
        return {"message": "ok", "name": opt.name, "status": "running"}


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
