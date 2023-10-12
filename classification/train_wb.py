import os
import sys
from pathlib import Path
import yaml
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

sys.path.append(str(Path(__file__).parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from train.service import TrainWrapper
import argparse


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    cfg = get_cfg(args)

    WRAPPER = TrainWrapper(
        cfg=cfg,
        num_workers=32,
    )

    model = EfficientLightning(
        model=WRAPPER.model,
        num2label=WRAPPER.num2label,
        batch_size=WRAPPER.batch_size,
        decay=WRAPPER.decay,
        augmentation=DataAugmentation,
        weights=WRAPPER.weights,
    )

    wandb_logger = WandbLogger(
        project=WRAPPER.cat,
        save_dir=WRAPPER.save_dir,
        name=WRAPPER.experiment_name,
        log_model=True,
        id=WRAPPER.experiment_name,
    )

    # Initialize a trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[1],
        max_epochs=cfg.epochs, # 60,
        precision=16,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=WRAPPER.get_callbacks(),
    )
    
    # Train the model ⚡
    trainer.fit(model, WRAPPER.train_loader, WRAPPER.val_loader)

    path_ = WRAPPER.get_best_model(model)


def parse_args():
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
    args = parser.parse_args()
    return args


def get_cfg(args: argparse.Namespace) -> EasyDict:
    cfg_path = args.cfg
    with open(cfg_path) as f:
        cfg = yaml.load(f, yaml.Loader)
    
    cfg.update(vars(args))
    return EasyDict(cfg)
    

if __name__ == "__main__":
    main()