from fastapi import Request, FastAPI, HTTPException
import asyncio
import traceback
import json
import subprocess
import os

import torch.multiprocessing as mp

from train import train_gpu
from options.train_options import TrainOptions
from data import create_dataset
from enum import Enum
from pydantic import create_model, BaseModel, Field


git_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__))
    .decode("ascii")
    .strip()
)
print("Launching JoliGAN Server\ncommit=%s" % git_hash)

description = """This is the JoliGAN server API documentation.
"""
app = FastAPI(title="JoliGAN server", description=description)

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


def stop_training(context):
    for process in context.processes:
        process.terminate()

    try:
        context.join()
    except Exception as e:
        print(e)


def is_alive(context):
    alive = True

    for process in context.processes:
        if not process.is_alive():
            alive = False

    return alive


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

    world_size = len(opt.gpu_ids)
    dataset = create_dataset(opt)

    ctx[name] = mp.spawn(
        train_gpu,
        args=(
            world_size,
            opt,
            dataset,
        ),
        nprocs=world_size,
        join=False,
    )

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
        return {"status": "running", "name": name}
    else:
        raise HTTPException(status_code=404, detail="Not found")


@app.get("/train", status_code=200, summary="Get the status of all training processes")
async def get_train_processes():
    processes = []
    for name in ctx:
        processes.append({"status": "running", "name": name})

    return {"processes": processes}


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
