import sys
import os
import logging
import ultralytics

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.utils import  yolo_proc_for_img_gen
from train.datasets import InferDataset
from train.augmentation import PreProcess

logging.getLogger('ultralytics').handlers = [logging.NullHandler()]

Pre = PreProcess(gray=False, vflip=False, arch="eff")
yolo_path = '/home/achernikov/CLS/best_people_28092023.pt'
src_path = '/home/achernikov/DATA/datasets/tits_size/picture'
dst_path = '/home/achernikov/DATA/segmentation_tits_size'
yolo_proc_for_img_gen(yolo_path, InferDataset, Pre, src_path, dst_path)

