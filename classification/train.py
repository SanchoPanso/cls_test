from datetime import timedelta
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Union

sys.path.append(str(Path(__file__).parent))
from engine.trainer import Trainer
from engine.wrapper import TrainWrapper
from utils.cfg import get_opts


def main():
    torch.cuda.empty_cache()

    args = parse_args()
    opts = get_opts(args, args.cfg)
    
    wrapper = TrainWrapper(opts)
    model = wrapper.get_model()
    
    # Initialize a trainer
    trainer = Trainer(wrapper)
    
    # Train the model ⚡
    trainer.fit(model)
    

def parse_args(src_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default=None,
        help="Path to configuration file with data structure description",
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--cat", type=str, default="hair_type", help="Category for training")
    parser.add_argument("--bg_cat", type=str, default="background", help='Background category')
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--mode", type=str, default="train", choices=['train', 'all'],
        help="Splitting mode (train - 80 percent train, 20 percent val,"\
             "all - 100 percent train, 100 percent val)"
    )
    parser.add_argument("--epochs", type=int, default=1, help='Number of epochs')
    parser.add_argument("--decay", type=float, default=0.001,help="Decay param for Adam optimizer")
    parser.add_argument("--arch", type=str, default="eff", help="Model architecture")
    parser.add_argument("--gray", type=bool, default=False, help="Gray scale")
    parser.add_argument(
        "--vflip",
        type=bool,
        default=True,
        help="vflip augmentation",
        required=False,
    )
    
    parser.add_argument(
        "--badlist_path",
        type=str,
        default=None, # os.path.join(os.path.dirname(__file__), 'data/badlist.txt'),
        help='Path to txt file which is a list of bad annotatated images',
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None, # 'body_type/v__6_train_eff_48_0.2/checkpoints/epoch=31-step=12320.ckpt',#None,
    )
    
    args = parser.parse_args(src_args)
    return args
    

if __name__ == "__main__":
    main()