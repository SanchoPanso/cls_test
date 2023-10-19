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
# from train.datasets import InferDataset
# from train.augmentation import PreProcess
from utils.cfg_handler import get_cfg
from utils.utils import apply_bilinear_interp, read_dataset_data

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    cfg = get_cfg()
    args = parse_args()
    
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    masks_dir = os.path.join(cfg['data_path'], cfg['masks2_dir'])
    datasets_dir = os.path.join(cfg['data_path'], cfg['datasets_dir'])
    result_dir = args.result_dir
    
    #yolo_proc_for_img_gen(args.model_path, images_dir, result_dir, args.process_all, group_img_list)
    if args.mode == 'segment':
        create_segments(args.model_path, images_dir, masks_dir, args.process_all)
    else:
        create_masks(images_dir, masks_dir, datasets_dir, result_dir, ['tits_size', 'sex_position'])

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='mask', choices=['segment', 'mask'])
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/best_people_28092023.pt')
    parser.add_argument('--groups', type=str, nargs='*', default=None)
    parser.add_argument('--process_all', action='store_true')
    parser.add_argument('--only_checked', action='store_true')
    parser.add_argument('--mark_checked', action='store_true')
    parser.add_argument('--result_dir', type=str, default='/home/achernikov/CLS/TEST/masks2')#None)
    
    args = parser.parse_args()
    return args


def create_masks(images_dir, masks_dir, datasets_dir, result_dir, groups, only_checked=False, only_unique=True):
    
    experiment_name = f"v__{len(os.listdir(result_dir))}"
    os.makedirs(os.path.join(result_dir, experiment_name), exist_ok=True)
    seen_names = set()   

    if groups is None:
        group = 'all'
        group_dir = os.path.join(result_dir, experiment_name, group)
        os.makedirs(group_dir, exist_ok=True)
        create_masks_set(images_dir, masks_dir, group_dir, only_checked)  
        return
    
    for group in groups:
        group_dir = os.path.join(result_dir, experiment_name, group)
        print(group_dir)
        os.makedirs(group_dir, exist_ok=True)
        
        dataset_path = os.path.join(datasets_dir, group + '.json')
        dataset_data = read_dataset_data(dataset_path)
        group_img_fns = dataset_data['path'].tolist()
        group_names = set(map(lambda x: os.path.splitext(x), group_img_fns))
        
        seen_names = create_masks_set(images_dir, masks_dir, group_dir, only_checked, group_names, seen_names)    

        if not only_unique:
            seen_names = set()
        


def create_masks_set(images_dir, masks_dir, dst_dir, only_checked=False, group_names=None, seen_names=None):
    img_name2fn = {os.path.splitext(fn)[0]: fn for fn in os.listdir(images_dir)}
    masks_files = os.listdir(masks_dir) 
    seen_names = seen_names if seen_names else set()
    
    for mask_file in tqdm(masks_files):
        with open(os.path.join(masks_dir, mask_file)) as js_f:
            data = json.load(js_f)

        for i in data:
            segments = data[str(i)]['segments']
            if only_checked and data[str(i)]['status'] != 'unchecked':
                continue
            
            name, ext = os.path.splitext(mask_file)
            if name not in img_name2fn:
                continue
            
            if group_names is not None and name not in group_names:
                continue
            if len(seen_names) > 0 and name not in seen_names:
                continue
            
            seen_names.add(name)
            
            img = cv2.imread(os.path.join(images_dir, img_name2fn[name]))
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            
            for segment in segments:
                segment = np.array(segment).reshape(-1, 1, 2)
                segment[..., 0] *= img.shape[1]
                segment[..., 1] *= img.shape[0]
                segment = segment.astype('int32')
                cv2.fillPoly(mask, [segment], 255)
            
            img = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(os.path.join(dst_dir, f"{name}_{i}.jpg"), img)
        
    return seen_names


# def yolo_proc_for_img_gen(yolo_model_path, src_path, dst_path, process_all=False, group_img_list=None):
#     """This is version of 'yolo_proc' for creating dataset for image generation

#     :param yolo_model_path: path to yolo model
#     :param src_path: directory with source images
#     :param dst_path: directory in wcich subdirs 'background', 'female', 'male' will be created
#     """

#     yolo_model = YOLO(yolo_model_path)
#     os.makedirs(src_path, exist_ok=True)
#     os.makedirs(os.path.join(dst_path, 'female'), exist_ok=True)
#     os.makedirs(os.path.join(dst_path, 'male'), exist_ok=True)
#     os.makedirs(os.path.join(dst_path, 'background'), exist_ok=True)
#     os.makedirs(os.path.join(dst_path, 'boxes'), exist_ok=True)

#     list_of_paths = glob(os.path.join(src_path, '*'))
    
#     # Leave only images from dataset
#     if group_img_list is not None:
#         list_of_paths = [item for item in list_of_paths if os.path.basename(item) in group_img_list]
    
#     # Exclude already processed images
#     if not process_all:
#         list_2_intersect = glob(os.path.join(dst_path, 'boxes', '*'))
#         list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
#         list_of_paths = [
#             item
#             for item in list_of_paths
#             if item.split("/")[-1].split(".")[0] not in list_2_intersect
#         ]
        
#     # infer = InferDataset(list_of_paths, Pre)
#     # loader = DataLoader(infer, 40, num_workers=32, pin_memory=True, shuffle=False)
#     gender_dict = {"girl": "female", "man": "male"}
    
#     for img_path in tqdm(list_of_paths):
#         name, ext = os.path.splitext(os.path.split(img_path)[1])
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
        
#         yolo_results = yolo_model(img)
    
#         for image_id, result in enumerate(yolo_results):
#             yolo_meta_dict = {}
#             bg_img = img.copy()
    
#             for j in range(len(result.boxes.xyxy)):
#                 if result.boxes.conf[j] >= 0.75:
#                     bbox_tensor = result.boxes.xyxy[j]#.to(torch.float16)
#                     bbox_cls = result.boxes.cls[j].to(int)
#                     bbox_cls = gender_dict[result.names[int(bbox_cls)]]
                    
#                     x1, y1, x2, y2 = map(lambda x: x.int().item(), bbox_tensor.cpu())
#                     bg_img[y1: y2, x1: x2] = apply_bilinear_interp(bg_img[y1: y2, x1: x2])
                    
#                     mask = result.masks.data[j].cpu().unsqueeze(0).numpy()
#                     mask = (mask > 0.5).astype('uint8')[0]
#                     mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#                     mask = mask[..., np.newaxis]
                    
#                     masked_img = img * mask
#                     cv2.imwrite(os.path.join(dst_path, bbox_cls, f'{name}_{j}{ext}'), masked_img)
                    
#                     yolo_meta_dict[j] = {
#                         "conf": float(result.boxes.conf[j]),
#                         "bbox": bbox_tensor.tolist(),
#                         "cls": gender_dict[bbox_cls]
#                         if bbox_cls in gender_dict.keys()
#                         else bbox_cls,
#                     }
#             cv2.imwrite(os.path.join(dst_path, 'background', f'{name}{ext}'), bg_img)
                    
#             if len(yolo_meta_dict) > 0:
#                 with open(os.path.join(dst_path, 'boxes', f"{name}.json"), "w") as f:
#                     json.dump(yolo_meta_dict, f)
                    
#     print("done")
    
    
    
def create_segments(yolo_model_path, src_path, dst_path, process_all=False, conf_thresh=0.75):
   
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
                
                yolo_meta_dict[j] = {
                    "cls": meta_cls,
                    "status": "unchecked",
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
