import os
import time
from celery import Celery
from typing import List
from cls.testing.pipeline import pipeline

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER")
celery.conf.result_backend = os.environ.get("CELERY_BACKEND")


@celery.task(name="create_task")
def create_task(guids: List[str], yolo_path: str, cls_inference_type: str):
    pipeline(guids, yolo_path, cls_inference_type)
    return True
