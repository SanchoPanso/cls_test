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
from sqlalchemy.sql import text

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.utils.database import update_picture_by_path
from cls.classification.utils.database import Picture
import cls.classification.utils.postgres_db as pdb
from cls.classification.utils.postgres_db import PostgreSQLHandler
from cls.classification.engine.options import OptionParser

LOGGER = logging.getLogger(__name__)
logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    args = parse_args()
    create_segments(args.model_path, args.pictures_dir, args.segments_dir, 
                    True, # args.process_all, 
                    args.mark_approved)


def parse_args():
    parser = OptionParser()
    
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/people_models/best_people_28092023.pt')
    parser.add_argument('--process_all', action='store_true', default=False)
    parser.add_argument('--mark_approved', action='store_true', default=False)
    
    args = parser.parse_args()
    return args
        
    
def create_segments(yolo_model_path, src_path, dst_path, 
                    process_all=False, mark_approved=False, conf_thresh=0.75):
   
    os.makedirs(dst_path, exist_ok=True)
    yolo_model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]
    yolo_model = YOLO(yolo_model_path).to('cuda')
    list_of_paths = glob(os.path.join(src_path, '*'))
    
    # Exclude already processed images, if we dont want to process all images
    if not process_all:
        list_2_intersect = glob(os.path.join(dst_path, '*'))
        list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
        list_of_paths = [
            item
            for item in list_of_paths
            if item.split("/")[-1].split(".")[0] not in list_2_intersect
        ]
    
    db_handler = None   # PostgreSQLHandler()
            
    for img_path in tqdm(list_of_paths):
        create_image_segments(
            img_path, 
            dst_path, 
            yolo_model,
            db_handler, 
            yolo_model_name, 
            conf_thresh, 
            mark_approved
        )

    LOGGER.info("Done")


def create_image_segments(
    img_path: str,
    dst_path: str, 
    yolo_model: YOLO, 
    db_handler: PostgreSQLHandler or None,
    yolo_model_name: str,
    conf_thresh: float, 
    mark_approved: bool):
    
    gender_dict = {"girl": "female", "man": "male"}    
    name, ext = os.path.splitext(os.path.basename(img_path))
    img = cv2.imread(img_path)
    if img is None:
        return
    
    yolo_results = yolo_model(img[:, :, ::-1], stream=True)

    for result in yolo_results:
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
            
            # Update database
            picture_info = Picture(
                path=os.path.basename(img_path), 
                model_version=yolo_model_name, 
                status=status
            )
            update_picture_by_path(picture_info)
            
            yolo_meta_dict[j] = {
                "cls": meta_cls,
                "conf": conf,
                "bbox": bbox,
                "segments": segments,
            }
        
        if db_handler is not None and db_handler.select_picture_by_path(os.path.basename(img_path)) is None:
            
            status = 'approved' if mark_approved else 'unchecked'
            picture_info = pdb.Picture(
                path=os.path.basename(img_path), 
                model_version=yolo_model_name, 
                status=status,
                segments=yolo_meta_dict,
            )
            db_handler.update_picture_by_path(picture_info)
            print(img_path)
        
        with open(os.path.join(dst_path, f"{name}.json"), "w") as f:
            json.dump(yolo_meta_dict, f)


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

# ### sync db (https://confluence.atlassian.com/confkb/error-when-publishing-pages-error-duplicate-key-value-violates-unique-constraint-tablename-_pkey-1167824619.html)
# with db_handler.engine.connect() as con:
#     rs = con.execute(text('SELECT last_value FROM pictures_ID_seq'))
#     for row in rs:
#         last_value = row[0]
# with db_handler.engine.connect() as con:
#     rs = con.execute(text('SELECT max(id) FROM pictures'))
#     for row in rs:
#         max_id = row[0]
# if last_value <= max_id:
#     with db_handler.engine.connect() as con:
#         rs = con.execute(text(f'ALTER SEQUENCE pictures_ID_seq RESTART WITH {max_id + 1}'))
# ######
