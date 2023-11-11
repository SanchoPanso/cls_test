import os
import sys
from pathlib import Path
import yaml
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from typing import Sequence

sys.path.append(str(Path(__file__).parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from classification.train.wrapper import TrainWrapper
import argparse


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    cfg = get_cfg(args)

    WRAPPER = TrainWrapper(
        cfg=cfg,
        num_workers=32,
    )
    
    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(
        "/home/achernikov/CLS/epoch=18-step=12027.pt",
        "cuda",
        _extra_files=extra_files,
    )
    print(model.num2label)

    model = EfficientLightning(
        model=model,
        num2label=model.num2label,
        augmentation=torch.nn.Sequential,
    )
    
    # if cfg.pretrained:
    #     checkpoint = torch.load(cfg.pretrained)
    #     state_dict = checkpoint['state_dict']
    #     model.load_state_dict(state_dict)

    # Initialize a trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.gpu],
        max_epochs=cfg.epochs, # 60,
        precision=16,
        log_every_n_steps=1,
        callbacks=WRAPPER.get_callbacks(),
    )

    trainer.validate(model, WRAPPER.val_loader)
    

def parse_args(src_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Type path to: model, json, data(optional)"
    )
    parser.add_argument(
        "--cfg", type=str, default=os.path.join(os.path.dirname(__file__), 'cfg', 'default.yaml'),
        help="Path to configuration file with data paths",
    )
    parser.add_argument(
        "--cat", dest="cat", type=str, 
        # default="tits_size", 
        default="body_type", 
        help="category", required=False,
    )
    parser.add_argument(
        "--bg_cat", type=str, 
        default="background", 
    )
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=48,
        help="Batch size",
        required=False,
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
        help="Callback mode",
        required=False,
    )
    parser.add_argument(
        "--decay",
        dest="decay",
        type=float,
        default=0.2,
        help="Decay",
        required=False,
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        type=str,
        default="eff",
        help="Model arch",
        required=False,
    )
    parser.add_argument(
        "--gray",
        dest="gray",
        type=bool,
        default=False,
        help="Gray scale",
        required=False,
    )
    parser.add_argument(
        "--vflip",
        dest="vflip",
        type=bool,
        default=True,
        help="vflip aug",
        required=False,
    )
    parser.add_argument(
        "--masks",
        dest="masks",
        type=bool,
        default=False,
        help="Using masks instead regular images",
        required=False,
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help='Number of epochs',
    )
    
    parser.add_argument(
        "--badlist_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data/badlist.txt'),
        help='Path to txt file which is a list of bad annotatated images',
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default='/home/achernikov/CLS/DATA/models/body_type/v__10_train_eff_48_0.2/checkpoints/epoch=65-step=25410.ckpt',#None,
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    
    args = parser.parse_args(src_args)
    return args


def get_cfg(args: argparse.Namespace) -> EasyDict:
    cfg_path = args.cfg
    with open(cfg_path) as f:
        cfg = yaml.load(f, yaml.Loader)
    
    cfg.update(vars(args))
    return EasyDict(cfg)
    

if __name__ == "__main__":
    main()