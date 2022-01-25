from fastapi import Request, FastAPI, HTTPException
import asyncio
import traceback
import json

import torch.multiprocessing as mp

from train import train_gpu
from options.train_options import TrainOptions
from data import create_dataset
from enum import Enum
from pydantic import create_model, BaseModel


description = """This is the JoliGAN server API documentation.
"""

app = FastAPI(
    title="JoliGAN server",
    description = description
)

# Additional schema
TrainOptionsSchema = TrainOptions().get_schema()
# Ensure schema is valid at startup
json.dumps(
    TrainOptionsSchema,
    ensure_ascii=False,
    indent=None,
    separators=(",", ":"),
).encode("utf-8")

generic_openapi = app.openapi
def custom_openapi():
    if not app.openapi_schema:
        app.openapi_schema = generic_openapi()
        app.openapi_schema["components"]["schemas"]["TrainOptions"] = TrainOptionsSchema
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
    "/train/{name}", status_code=201,
    summary="Start a training process with given name.",
    description="The training process will be created using the same options as command line",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": { "$ref": "#/components/schemas/TrainOptions" }
                }
            }
        }
    }
)
async def train(name : str, request : Request):
    train_options = await request.json()

    try:
        opt = TrainOptions().parse_json(train_options)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="{0}".format(e))

    world_size = len(opt.gpu_ids)
    dataset = create_dataset(opt)

    ctx[name] = mp.spawn(train_gpu,
                        args=(world_size,opt,dataset,),
                        nprocs=world_size,
                        join=False)
    return { "message": "ok", "name" : name }

@app.get("/train/{name}", status_code=200, summary="Get the status of a training process")
async def get_train(name : str):
    if name in ctx:
        status = "running" if is_alive(ctx[name]) else "stopped"
        return {"status" : "running", "name": name}
    else:
        raise HTTPException(status_code = 404, detail="Not found")

@app.get("/train", status_code=200, summary="Get the status of all training processes")
async def get_train_processes():
    processes = []
    for name in ctx:
        processes.append({"status": "running", "name": name})

    return { "processes": processes }

@app.delete(
    "/train/{name}", status_code=200,
    summary = "Delete a training process.",
    description = "If the process is running, it will be stopped."
)
async def delete_train(name : str):
    if name in ctx:
        stop_training(ctx[name])
        del ctx[name]
        return { "message":"ok", "name": name}
    else:
        raise HTTPException(status_code = 404, detail="Not found")

