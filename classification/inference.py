import os
import sys
import argparse
from pathlib import Path
import yaml
import json
import cv2
import pandas as pd
import shutil
import torch
import numpy as np
from tqdm import tqdm
from io import StringIO
from easydict import EasyDict
from pytorch_lightning import Trainer
from typing import Sequence
from torch.utils.data import DataLoader
from torch import nn

sys.path.append(str(Path(__file__).parent.parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning, ModelBuilder
from train.service import TrainWrapper
from train.datasets import InferenceDirDataset, InferenceContourDataset
from utils.cfg_handler import get_cfg
from utils.utils import read_dataset_data
from utils.logger import get_logger

LOGGER = get_logger(__name__)

def main():
    torch.cuda.empty_cache()

    args = parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))
    LOGGER.info(f'Configuration: {cfg}')

    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(args.model, args.device, _extra_files=extra_files)
    num2label = json.loads(extra_files['num2label.txt'])
    LOGGER.info(f'num2label = {num2label}')
    
    if args.group:
        df = read_dataset_data(os.path.join(cfg.data_path, cfg.datasets_dir, f'{args.group}.json'))
        images_dir = os.path.join(cfg.data_path, cfg.images_dir) 
        segments_dir = os.path.join(cfg.data_path, cfg.segments_dir)
        ds = InferenceContourDataset(images_dir, segments_dir, df)
        source_is_ready = False
        LOGGER.info(f'group = {args.group}')
        LOGGER.info('You have chosen the algorithm that infer the specific group.'
                    'Reading original images and json segments will be used instead of using ready source images')
    
    else:
        src_dirs = [args.source]
        ds = InferenceDirDataset(src_dirs)
        source_is_ready = True
        LOGGER.info(f'source = {args.source}')
        
    
    model = model.eval()
    
    inference_dir = os.path.join(cfg.test_path, cfg.inference_dir)
    os.makedirs(inference_dir, exist_ok=True)
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"v__{len(os.listdir(inference_dir))}"
        
    save_dir = os.path.join(inference_dir, experiment_name)
    LOGGER.info(f'save_dir = {save_dir}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for num in num2label:
        os.makedirs(os.path.join(save_dir, num2label[num]), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'trash'), exist_ok=True)
    
    infer(model, ds, args.batch, save_dir, num2label, source_is_ready, args.conf_thresh)


def infer(model, ds, batch, save_dir: str, num2label: dict, source_is_ready, conf_thresh: float, use_softmax=True):
    loader = DataLoader(ds, batch)
    sigmoid = torch.nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    progress_bar = tqdm(total=len(ds))
    
    for elem in loader:
        if source_is_ready:
            img, paths = elem
        else:
            img, masks, img_path, mask_fn = elem
            
        with torch.no_grad():
            logits = model(img.to(torch.float32).to('cuda:0'))
            if use_softmax:
                logits = softmax(logits)
            else:
                logits = sigmoid(logits)
        
        logits = logits.cpu().numpy()
        for i, lgt in enumerate(logits):
            pred_class_id = lgt.argmax()
            if lgt[pred_class_id] < conf_thresh:
                pred_class_name = 'trash'
            else:
                pred_class_name = num2label[str(pred_class_id)]
            
            if source_is_ready:
                path = paths[i]
                fn = os.path.basename(path)
                dst_path = os.path.join(save_dir, pred_class_name, fn)
                shutil.copy(path, dst_path)
            else:
                img0 = cv2.imread(img_path[i])
                mask = cv2.resize(masks[i].cpu().numpy(), (img0.shape[1], img0.shape[0]))
                img0 = cv2.bitwise_and(img0, img0, mask=mask)
                dst_path = os.path.join(save_dir, pred_class_name, mask_fn[i])
                cv2.imwrite(dst_path, img0)
            
            progress_bar.update(1)
            progress_bar.set_postfix(path=os.path.basename(dst_path))


def parse_args(src_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Type path to: model, json, data(optional)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        # default='/home/achernikov/CLS/DATA/models/tits_size_with_trash/v__1_train_eff_softmax_32_0.001/checkpoints/epoch=22-step=14950.pt',
        default='/home/achernikov/CLS/DATA/models/body_type2_with_trash/v__4_train_eff_softmax_32_0.001/checkpoints/epoch=66-step=45292.pt',
        # default='/home/achernikov/CLS/DATA/models/body_type2/v__1_all_eff_32_0.001/checkpoints/epoch=47-step=39936.pt',
    )
    parser.add_argument(
        "--cfg", type=str, default=os.path.join(os.path.dirname(__file__), 'cfg', 'default.yaml'),
        help="Path to configuration file with data paths",
    )
    parser.add_argument(
        "--group", type=str, 
        #default="body_type", 
        default="test", 
    )
    parser.add_argument(
        "--source", type=str, 
        default='/home/achernikov/CLS/MARKUP/masks/test/v__0',#None, 
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
    parser.add_argument('--experiment_name', type=str, default=None)
    
    args = parser.parse_args(src_args)
    return args
    

if __name__ == "__main__":
    main()