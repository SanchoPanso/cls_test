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
from classification.utils.cfg import get_cfg
from classification.utils.general import read_dataset_data

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    cfg = get_cfg()
    args = parse_args()
    
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    segments_dir = os.path.join(cfg['data_path'], cfg['segments_dir'])
    masks_dir = os.path.join(cfg['markup_path'], cfg['masks_dir'])
    datasets_dir = os.path.join(cfg['data_path'], cfg['datasets_dir'])
    
    group = args.group
    process_all = args.process_all
    experiment_name = args.experiment_name
    
    dataset_path = os.path.join(datasets_dir, group + '.json')
    dataset_data = read_dataset_data(dataset_path)
    group_img_fns = dataset_data['path'].tolist()
    group_names = set(map(lambda x: os.path.splitext(x)[0], group_img_fns))
    
    class_names = dataset_data.columns[1:]
    names2cls = {}
    for i in range(len(dataset_data)):
        row = dataset_data.iloc[i]
        name = os.path.splitext(row['path'])[0]
        if row[class_names].values.max() == 0:
            cls = 'trash'
        else:
            cls = class_names[row[class_names].values.argmax()]
        names2cls[name] = cls
    
                
    tist_size_path = os.path.join(datasets_dir, 'tits_size.json')
    dataset_data = read_dataset_data(tist_size_path)
    tits_size_img_fns = dataset_data['path'].tolist()
    tits_size_names = set(map(lambda x: os.path.splitext(x)[0], tits_size_img_fns))
    
    body_type_path = os.path.join(datasets_dir, 'body_type.json')
    dataset_data = read_dataset_data(body_type_path)
    body_type_img_fns = dataset_data['path'].tolist()
    body_type_names = set(map(lambda x: os.path.splitext(x)[0], body_type_img_fns))

    group_names = group_names - tits_size_names - body_type_names
    
    for name in group_names:
        assert name not in body_type_names
        assert name not in tits_size_names
        
    group_dir = os.path.join(masks_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    if experiment_name is None:
        experiment_name = f"v__{len(os.listdir(group_dir))}"
    
    result_dir = os.path.join(group_dir, experiment_name) 
    os.makedirs(result_dir, exist_ok=True)
    for name in class_names:
        os.makedirs(os.path.join(result_dir, name), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'trash'), exist_ok=True)
    
    create_masks_set(images_dir, segments_dir, result_dir, group_names, process_all, names2cls)      


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--group', type=str, default='hair_type')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--process_all', action='store_true')
    # parser.add_argument('--only_checked', action='store_true') # TODO
    
    args = parser.parse_args()
    return args


def create_masks_set(images_dir, segments_dir, dst_dir, group_names=None, process_all=False, names2cls=None):
    img_name2fn = {os.path.splitext(fn)[0]: fn for fn in os.listdir(images_dir)}
    masks_files = os.listdir(segments_dir) 
    
    result_names = []
    for mask_file in masks_files:
        name, ext = os.path.splitext(mask_file)
        if name not in img_name2fn:
            continue
        if group_names is not None and name not in group_names:
            continue
            
        result_names.append(name)
        
    
    for name in tqdm(result_names):
        with open(os.path.join(segments_dir, name + '.json')) as js_f:
            data = json.load(js_f)

        for i in data:
            cls = names2cls[name]
            save_mask_path = os.path.join(dst_dir, cls, f"{name}_{i}.jpg")
            # if not process_all and os.path.exists(save_mask_path): # TODO
            #     continue
            
            img = cv2.imread(os.path.join(images_dir, img_name2fn[name]))
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            segments = data[str(i)]['segments']
            
            for segment in segments:
                segment = np.array(segment).reshape(-1, 1, 2)
                segment[..., 0] *= img.shape[1]
                segment[..., 1] *= img.shape[0]
                segment = segment.astype('int32')
                cv2.fillPoly(mask, [segment], 255)
            
            img = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(save_mask_path, img)
        



if __name__ == '__main__':
    main()
