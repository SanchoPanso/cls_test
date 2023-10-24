import sys
import os
from glob import glob
import cv2
from tqdm import tqdm
import logging
import numpy as np
import json
from ultralytics import YOLO
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
# from train.datasets import InferDataset
# from train.augmentation import PreProcess
from utils.cfg_handler import get_cfg

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    cfg = get_cfg()
    args = parse_args()
    
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    segments_dir = os.path.join(cfg['data_path'], cfg['segments_dir'])
    
    create_segments(args.model_path, images_dir, segments_dir, args.process_all, args.mark_approved)
   
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/best_people_28092023.pt')
    parser.add_argument('--process_all', action='store_true')
    parser.add_argument('--mark_approved', action='store_true')
    
    args = parser.parse_args()
    return args
        
    
def create_segments(yolo_model_path, src_path, dst_path, 
                    process_all=False, mark_approved=False, conf_thresh=0.75):
   
    os.makedirs(dst_path, exist_ok=True)
    yolo_model = YOLO(yolo_model_path).to('cuda')
    list_of_paths = glob(os.path.join(src_path, '*'))
    
    # Exclude already processed images
    if not process_all:
        list_2_intersect = glob(os.path.join(dst_path, '*'))
        list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
        list_of_paths = [
            item
            for item in list_of_paths
            if item.split("/")[-1].split(".")[0] not in list_2_intersect
        ]
    
    gender_dict = {"girl": "female", "man": "male"}    
    
    for img_path in tqdm(list_of_paths):
        name, ext = os.path.splitext(os.path.split(img_path)[1])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        yolo_results = yolo_model(img, stream=True)
    
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
                
                yolo_meta_dict[j] = {
                    "cls": meta_cls,
                    "status": status,
                    "conf": conf,
                    "bbox": bbox,
                    "segments": segments,
                }
            
            with open(os.path.join(dst_path, f"{name}.json"), "w") as f:
                json.dump(yolo_meta_dict, f)

    print("done")


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
