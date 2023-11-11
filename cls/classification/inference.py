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
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.engine.augmentation import DataAugmentation
from cls.classification.engine.datasets import InferenceDirDataset, InferenceContourDataset
from cls.classification.engine.options import OptionParser
from cls.classification.utils.general import read_dataset_data

LOGGER = logging.getLogger(__name__)

def main():
    args = parse_args()
    
    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(args.model, args.device, _extra_files=extra_files)
    model = model.eval()
    num2label = json.loads(extra_files['num2label.txt'])
    LOGGER.info(f'num2label: {num2label}')
    
    inference_dir = args.inference_dir
    os.makedirs(inference_dir, exist_ok=True)
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"v__{len(os.listdir(inference_dir))}"
        
    save_dir = os.path.join(inference_dir, experiment_name)
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for num in num2label:
        os.makedirs(os.path.join(save_dir, num2label[num]), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'trash'), exist_ok=True)
    
    augmentation = DataAugmentation()
    
    if args.group:
        df = read_dataset_data(os.path.join(args.datasets_dir, f'{args.group}.json'))
        image_filenames = df['path'].tolist()
        images_dir = args.pictures_dir 
        segments_dir = args.segments_dir
        dataset = InferenceContourDataset(images_dir, segments_dir, image_filenames, augmentation)
        
        LOGGER.info(f'Group: {args.group}')
        LOGGER.info(f'Result will be saved in \033[1m{save_dir}\033[0m')
        infer_with_contours(model, dataset, args.batch, save_dir, num2label, args.conf_thresh)
                
    else:
        src_dirs = [args.source]
        dataset = InferenceDirDataset(src_dirs, transforms=augmentation)
        
        LOGGER.info(f'Source: {args.source}')
        LOGGER.info(f'Result will be saved in \033[1m{save_dir}\033[0m')
        infer_with_data(model, dataset, args.batch, save_dir, num2label, args.conf_thresh) 
    
    LOGGER.info('Done')   
    
    
def infer_with_data(model, dataset, batch, save_dir, num2label, conf_thresh):
    loader = DataLoader(dataset, batch)
    sigmoid = torch.nn.Sigmoid()
    # softmax = nn.Softmax(dim=1)
    progress_bar = tqdm(total=len(dataset))
    
    for elem in loader:
        img, paths = elem
            
        with torch.no_grad():
            logits = model(img.to(torch.float32).to('cuda:0'))
            logits = sigmoid(logits)
            logits = logits.cpu().numpy()

        for i, lgt in enumerate(logits):
            pred_class_id = lgt.argmax()
            
            if lgt[pred_class_id] < conf_thresh:
                pred_class_name = 'trash'
            else:
                pred_class_name = num2label[str(pred_class_id)]
            
            path = paths[i]
            fn = os.path.basename(path)
            dst_path = os.path.join(save_dir, pred_class_name, fn)
            shutil.copy(path, dst_path)
            
            progress_bar.update(1)
            progress_bar.set_postfix(path=os.path.basename(dst_path))


def infer_with_contours(model, dataset, batch, save_dir, num2label, conf_thresh):
    loader = DataLoader(dataset, batch)
    sigmoid = torch.nn.Sigmoid()
    # softmax = nn.Softmax(dim=1)
    progress_bar = tqdm(total=len(dataset))
    
    for elem in loader:
        img, segments_reprs, img_path, mask_fn = elem
            
        with torch.no_grad():
            logits = model(img.to(torch.float32).to('cuda:0'))
            logits = sigmoid(logits)
            logits = logits.cpu().numpy()
        
        for i, lgt in enumerate(logits):
            pred_class_id = lgt.argmax()
            
            if lgt[pred_class_id] < conf_thresh:
                pred_class_name = 'trash'
            else:
                pred_class_name = num2label[str(pred_class_id)]
            
            img0 = cv2.imread(img_path[i])
            segments = json.loads(segments_reprs[i])['segments']
            img0, _ = get_segmented_img(img0, segments)
            dst_path = os.path.join(save_dir, pred_class_name, mask_fn[i])
            cv2.imwrite(dst_path, img0)
        
            progress_bar.update(1)
            progress_bar.set_postfix(path=os.path.basename(dst_path))


def parse_args(src_args: Sequence[str] | None = None):
    parser = OptionParser()
    
    parser.add_argument(
        "--model",
        type=str,
        default='/home/achernikov/CLS/DATA/models/hair_type/v__3_all_eff_48_0.001/checkpoints/epoch=32-step=15543.pt',
        # default='/home/achernikov/CLS/DATA/models/tits_size_with_trash/v__1_train_eff_softmax_32_0.001/checkpoints/epoch=22-step=14950.pt',
        # default='/home/achernikov/CLS/DATA/models/body_type2_with_trash/v__4_train_eff_softmax_32_0.001/checkpoints/epoch=66-step=45292.pt',
        # default='/home/achernikov/CLS/DATA/models/body_type2/v__1_all_eff_32_0.001/checkpoints/epoch=47-step=39936.pt',
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


def get_segmented_img(img, segments):
    mask = np.zeros(img.shape[:2], dtype='uint8')
    
    for segment in segments:
        segment = np.array(segment)
        segment = segment.reshape(-1, 1, 2)
        segment[..., 0] *= img.shape[1]
        segment[..., 1] *= img.shape[0]
        segment = segment.astype('int32')
        cv2.fillPoly(mask, [segment], 255)
    
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img, mask


if __name__ == "__main__":
    main()