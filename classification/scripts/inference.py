import os
import sys
import argparse
from pathlib import Path
import yaml
import json
import pandas as pd
import shutil
import torch
from io import StringIO
from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from typing import Sequence
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning, ModelBuilder
from train.service import TrainWrapper
from train.datasets import InferenceDataset
from utils.cfg_handler import get_cfg
from utils.utils import read_dataset_data

def main():
    torch.cuda.empty_cache()

    args = parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(
        args.model,
        args.device,
        _extra_files=extra_files,
    )
    num2label =json.loads(extra_files['num2label.txt'])
    print(num2label)
    
    df = read_dataset_data(os.path.join(cfg.data_path, cfg.datasets_dir, 'test.json'))
    src_dirs = [os.path.join(cfg.data_path, cfg.masks_dir, 'female'),
                os.path.join(cfg.data_path, cfg.masks_dir, 'male')]
    ds = InferenceDataset(df, src_dirs)
    
    model = model.eval()
    loader = DataLoader(ds, args.batch)
    
    infer_dir = os.path.join(cfg.test_path, cfg.infer_dir)
    os.makedirs(infer_dir, exist_ok=True)
    for num in num2label:
        os.makedirs(os.path.join(infer_dir, num2label[num]), exist_ok=True)
    os.makedirs(os.path.join(infer_dir, 'trash'), exist_ok=True)
    
    sigmoid = torch.nn.Sigmoid()
    for elem in loader:
        img, paths = elem
        
        with torch.no_grad():
            logits = model(img.to(torch.float32).to('cuda:0'))
            logits = sigmoid(logits)
        
        logits = logits.cpu().numpy()
        for i, lgt in enumerate(logits):
            path = paths[i]
            fn = os.path.basename(path)
            pred_class_id = lgt.argmax()
            
            if lgt[pred_class_id] < args.conf_thresh:
                pred_class_name = 'trash'
            else:
                pred_class_name = num2label[str(pred_class_id)]
            
            dst_path = os.path.join(infer_dir, pred_class_name, fn)
            print(dst_path)
            
            shutil.copy(path, dst_path)


def parse_args(src_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Type path to: model, json, data(optional)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cfg", type=str, default=os.path.join(os.path.dirname(__file__), 'cfg', 'default.yaml'),
        help="Path to configuration file with data paths",
    )
    parser.add_argument(
        "--cat", dest="cat", type=str, 
        default="body_type", 
        # default="test", 
        help="category", required=False,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size",
        required=False,
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )
    
    args = parser.parse_args(src_args)
    return args
    

if __name__ == "__main__":
    main()