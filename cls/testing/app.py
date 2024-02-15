from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cls.testing.pipeline import pipeline

YOLO_PATH = '/home/achernikov/CLS/people_models/best_people.pt'
app = FastAPI()


class Item(BaseModel):
    guids: List[str]


@app.post('/post_trained/')
def post_trained(item: Item):
    try:
        raise ValueError("aaa")
        pipeline(item.guids, YOLO_PATH)
    except Exception as e:
        raise HTTPException(502, str(e))
    
    return item


@app.get('/test/')
def test():
    return {"status": "OK"}
