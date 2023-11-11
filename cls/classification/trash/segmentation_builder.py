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

sys.path.append(str(Path(__file__).parent))
from classification.utils.general import  yolo_proc_for_img_gen
from classification.utils.cfg import get_cfg
from classification.utils.general import apply_bilinear_interp, read_dataset_data

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    cfg = get_cfg()
    args = parse_args()
    
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    masks_dir = os.path.join(cfg['data_path'], cfg['masks_dir'])
    result_dir = args.result_dir if args.result_dir else masks_dir
    
    if args.group:
        dataset_path = os.path.join(cfg['data_path'], cfg['datasets_dir'], args.group + '.json')
        dataset_data = read_dataset_data(dataset_path)
        group_img_list = dataset_data['path'].tolist()
    else:
        group_img_list = None
    
    yolo_proc_for_img_gen(args.model_path, images_dir, result_dir, args.process_all, group_img_list)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/best_people_28092023.pt')
    parser.add_argument('--group', type=str, default='')
    parser.add_argument('--process_all', action='store_true')
    parser.add_argument('--result_dir', type=str, default=None)
    
    args = parser.parse_args()
    return args


def yolo_proc_for_img_gen(yolo_model_path, src_path, dst_path, process_all=False, group_img_list=None):
    """This is version of 'yolo_proc' for creating dataset for image generation

    :param yolo_model_path: path to yolo model
    :param src_path: directory with source images
    :param dst_path: directory in wcich subdirs 'background', 'female', 'male' will be created
    """

    yolo_model = YOLO(yolo_model_path)
    os.makedirs(src_path, exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'female'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'male'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'background'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'boxes'), exist_ok=True)

    list_of_paths = glob(os.path.join(src_path, '*'))
    
    # Leave only images from dataset
    if group_img_list is not None:
        list_of_paths = [item for item in list_of_paths if os.path.basename(item) in group_img_list]
    
    # Exclude already processed images
    if not process_all:
        list_2_intersect = glob(os.path.join(dst_path, 'boxes', '*'))
        list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
        list_of_paths = [
            item
            for item in list_of_paths
            if item.split("/")[-1].split(".")[0] not in list_2_intersect
        ]
        
    # infer = InferDataset(list_of_paths, Pre)
    # loader = DataLoader(infer, 40, num_workers=32, pin_memory=True, shuffle=False)
    gender_dict = {"girl": "female", "man": "male"}
    
    for img_path in tqdm(list_of_paths):
        name, ext = os.path.splitext(os.path.split(img_path)[1])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        yolo_results = yolo_model(img)
    
        for image_id, result in enumerate(yolo_results):
            yolo_meta_dict = {}
            bg_img = img.copy()
    
            for j in range(len(result.boxes.xyxy)):
                if result.boxes.conf[j] >= 0.75:
                    bbox_tensor = result.boxes.xyxy[j]#.to(torch.float16)
                    bbox_cls = result.boxes.cls[j].to(int)
                    bbox_cls = gender_dict[result.names[int(bbox_cls)]]
                    
                    x1, y1, x2, y2 = map(lambda x: x.int().item(), bbox_tensor.cpu())
                    bg_img[y1: y2, x1: x2] = apply_bilinear_interp(bg_img[y1: y2, x1: x2])
                    
                    mask = result.masks.data[j].cpu().unsqueeze(0).numpy()
                    mask = (mask > 0.5).astype('uint8')[0]
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask = mask[..., np.newaxis]
                    
                    masked_img = img * mask
                    cv2.imwrite(os.path.join(dst_path, bbox_cls, f'{name}_{j}{ext}'), masked_img)
                    
                    yolo_meta_dict[j] = {
                        "conf": float(result.boxes.conf[j]),
                        "bbox": bbox_tensor.tolist(),
                        "cls": gender_dict[bbox_cls]
                        if bbox_cls in gender_dict.keys()
                        else bbox_cls,
                    }
            cv2.imwrite(os.path.join(dst_path, 'background', f'{name}{ext}'), bg_img)
                    
            if len(yolo_meta_dict) > 0:
                with open(os.path.join(dst_path, 'boxes', f"{name}.json"), "w") as f:
                    json.dump(yolo_meta_dict, f)
                    
    print("done")


if __name__ == '__main__':
    main()
