import sys
import os
import logging
import ultralytics
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.utils import  yolo_proc_for_img_gen
from train.datasets import InferDataset
from train.augmentation import PreProcess
from utils.cfg import get_cfg

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]


def main():
    cfg = get_cfg()
    args = parse_args()
    
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    masks_dir = os.path.join(cfg['data_path'], cfg['masks_dir'])
    
    Pre = PreProcess(gray=False, vflip=False, arch="eff")
    yolo_proc_for_img_gen(args.model_path, InferDataset, Pre, images_dir, masks_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/achernikov/CLS/best_people_28092023.pt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
