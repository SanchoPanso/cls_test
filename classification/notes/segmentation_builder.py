import sys

sys.path.append("/home/timssh/ML/TAGGING/CLS/classification")

from scripts.utils import  yolo_proc
from train.model import InferDataset
from train.augmentation import PreProcess

Pre = PreProcess(gray=False, vflip=False, arch="eff")
yolo_path = '/home/timssh/ML/TAGGING/CLS/instance/runs/segment/train7/weights/best.pt'
yolo_proc(yolo_path, InferDataset, Pre)