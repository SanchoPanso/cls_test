import sys
import os
from pathlib import Path
from glob import glob
import cv2
from tqdm import tqdm
import logging
import numpy as np
import json
from ultralytics import YOLO
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.utils.postgres_db import PostgreSQLHandler, Picture
from cls.classification.engine.options import OptionParser

LOGGER = logging.getLogger(__name__)
logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    args = parse_args()
    create_segments(args.model_path, args.process_all, args.mark_approved, args=args)


def parse_args():
    parser = OptionParser()
    
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/people_models/best_people.pt')
    parser.add_argument('--process_all', action='store_true', default=False)
    parser.add_argument('--mark_approved', action='store_true', default=False)
    
    args = parser.parse_args()
    return args
        
    
def create_segments(yolo_model_path, 
                    process_all=False, mark_approved=False, conf_thresh=0.5, args = None):
    
    src_path = args.pictures_dir
    yolo_model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]
    yolo_model = YOLO(yolo_model_path).to('cuda')
    list_of_image_paths = glob(os.path.join(src_path, '*'))

    db_handler = PostgreSQLHandler()
    
    # Exclude already processed images, if we dont want to process all images
    if not process_all:
        done_segments_path = db_handler.select_all_paths()
        list_of_image_paths = [item for item in list_of_image_paths if item not in done_segments_path]
    
    for img_path in tqdm(list_of_image_paths):
        create_image_segments(
            img_path, 
            yolo_model,
            db_handler, 
            yolo_model_name, 
            conf_thresh, 
            mark_approved
        )

    LOGGER.info("Done")


def create_image_segments(
    img_path: str,
    yolo_model: YOLO, 
    db_handler: PostgreSQLHandler | None,
    yolo_model_name: str,
    conf_thresh: float, 
    mark_approved: bool):
    
    gender_dict = {"girl": "female", "man": "male"}    
    img = cv2.imread(img_path)
    if img is None:
        return
    
    yolo_results = yolo_model(img[:, :, ::-1], stream=True)
    result = next(yolo_results)
    
    yolo_meta_dict = {}
    
    for j in range(len(result.boxes.xyxy)):
        if result.boxes.conf[j] < conf_thresh:
            continue
        
        bbox_tensor = result.boxes.xyxy[j]
        bbox_cls = result.boxes.cls[j].to(int)
        bbox_cls = gender_dict[result.names[int(bbox_cls)]]
        
        mask = result.masks.data[j].cpu().unsqueeze(0).numpy()
        mask = (mask > 0.5).astype('uint8')[0]                    
        segments = mask2segments(mask)
        
        bbox = bbox_tensor.tolist()
        conf = float(result.boxes.conf[j])
        meta_cls = gender_dict[bbox_cls] if bbox_cls in gender_dict else bbox_cls
        
        status = 'approved' if mark_approved else 'unchecked'
        
        yolo_meta_dict[j] = {
            "cls": meta_cls,
            "conf": conf,
            "bbox": bbox,
            "segments": segments,
        }
    
    status = 'approved' if mark_approved else 'unchecked'
    picture_info = Picture(
        path=os.path.basename(img_path), 
        model_version=yolo_model_name, 
        status=status,
        segments=yolo_meta_dict,
    )
    db_handler.update_picture_by_path(picture_info)
    print(img_path)


def mask2segments(mask: np.ndarray) -> list:
    height, width = mask.shape[:2]
    segments = []
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in contours:
        cnt = cnt.astype('float64')
        cnt[:, 0, 0] /= width
        cnt[:, 0, 1] /= height
        segment = cnt.reshape(-1).tolist()
        segments.append(segment)
    return segments


if __name__ == '__main__':
    main()

