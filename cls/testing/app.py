import traceback
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cls.testing.pipeline import set_preparing
from cls.testing.pipeline import pipeline
from cls.testing.worker import create_task

YOLO_PATH = 'people_models/best1.pt'
CLS_INFERENCE_TYPE = 'torchscript'
app = FastAPI()
app.mount("/static", StaticFiles(directory="cls/testing/static"), name="static")
templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    guids: List[str]


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", context={"request": request})


@app.post('/preparing_to_trained_async/')
def post_trained(item: Item):
    result = create_task.delay(item.guids, YOLO_PATH, CLS_INFERENCE_TYPE)
    return {"task_id": result.id}


@app.get("/tasks/{task_id}")
def get_status(task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JSONResponse(result)


@app.post('/set_preparing/{guid}')
def set_as_preparing(guid: str):
    set_preparing([guid], 'dev.')


@app.post('/preparing_to_trained/')
def post_trained(item: Item):
    try:
        pipeline(item.guids, YOLO_PATH, CLS_INFERENCE_TYPE)
    except Exception as exc:
        formatted_lines = traceback.format_exc().splitlines()
        raise HTTPException(502, formatted_lines[-1])
    
    return {"status": "OK", "guids": item.guids}


@app.get('/test/')
def test():
    return {"status": "OK"}
