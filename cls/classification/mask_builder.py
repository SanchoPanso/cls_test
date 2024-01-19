import sys
import os
from glob import glob
import cv2
from tqdm import tqdm
import logging
import numpy as np
import json
import argparse
import logging
from pathlib import Path

from cls.classification.engine.options import OptionParser
from cls.classification.engine.database import PostgreSQLHandler
from cls.classification.utils.general import read_dataset_data

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    
    group = args.group
    process_all = args.process_all
    experiment_name = args.experiment_name
    
    if group:
        dataset_path = os.path.join(args.datasets_dir, group + '.json')
        dataset_data = read_dataset_data(dataset_path)
        group_img_fns = dataset_data['path'].tolist()
        group_names = set(map(lambda x: os.path.splitext(x)[0], group_img_fns))
    else:
        group = 'all'
        group_names = None
    
    group_dir = os.path.join(args.masks_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    if experiment_name is None:
        experiment_name = f"v__{len(os.listdir(group_dir))}"
    
    result_dir = os.path.join(group_dir, experiment_name) 
    os.makedirs(result_dir, exist_ok=True)
    LOGGER.info(f"Result will be saved in {result_dir}")
    
    create_masks_set(args.pictures_dir, result_dir, group_names, process_all)      
    LOGGER.info(f"Done")
    

def parse_args():
    parser = OptionParser()
    
    parser.add_argument('--group', type=str, default='tits_size')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--process_all', action='store_true')
    # parser.add_argument('--only_checked', action='store_true') # TODO
    
    args = parser.parse_args()
    return args


def create_masks_set(images_dir, dst_dir, group_names=None, process_all=False):

    db_handler = PostgreSQLHandler()
    img_paths = os.listdir(images_dir)
    seg_paths = db_handler.select_all_paths()
    
    result_paths = []
    for seg_path in seg_paths:
        if seg_path not in img_paths:
            continue
        if group_names is not None and seg_path not in group_names:
            continue
            
        result_paths.append(seg_path)
        
    
    for path in tqdm(result_paths):
        name, ext = os.path.splitext(path)
        data = db_handler.select_picture_by_path(path).segments
        
        for i in data:
            save_mask_path = os.path.join(dst_dir, f"{name}_{i}.jpg")
            if not process_all and os.path.exists(save_mask_path): # TODO
                continue
            
            img = cv2.imread(os.path.join(images_dir, path))
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
