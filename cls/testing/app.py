import traceback
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cls.testing.pipeline import pipeline

YOLO_PATH = 'people_models/best1.pt'
app = FastAPI()


class Item(BaseModel):
    guids: List[str]


@app.post('/preparing_to_trained/')
def post_trained(item: Item):
    try:
        pipeline(item.guids, YOLO_PATH)
    except Exception as exc:
        formatted_lines = traceback.format_exc().splitlines()
        raise HTTPException(502, formatted_lines[-1])
    
    return {"status": "OK", "guids": item.guids}


@app.get('/test/')
def test():
    return {"status": "OK"}
