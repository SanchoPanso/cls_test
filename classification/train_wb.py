import os
import sys
import argparse
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from typing import Sequence

sys.path.append(str(Path(__file__).parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from train.service import TrainWrapper
from utils.cfg_handler import get_cfg


def main():
    torch.cuda.empty_cache()
    args = parse_args()
    cfg = get_cfg(args.cfg)
    cfg.update(vars(args))
    print('cfg:', cfg)

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
    
    if cfg.pretrained:
        cfg.pretrained = os.path.join(cfg.data_path, cfg.mask_dir, cfg.pretrained)
        checkpoint = torch.load(cfg.pretrained)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

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
        devices=[cfg.gpu],
        max_epochs=cfg.epochs, # 60,
        precision=16,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=WRAPPER.get_callbacks(),
    )

    # Train the model ⚡
    trainer.fit(model, WRAPPER.train_loader, WRAPPER.val_loader)

    path_ = WRAPPER.get_best_model(model)


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
        default=32,
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
        default=0.001,
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
        default=None, # os.path.join(os.path.dirname(__file__), 'data/badlist.txt'),
        help='Path to txt file which is a list of bad annotatated images',
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None, # 'body_type/v__6_train_eff_48_0.2/checkpoints/epoch=31-step=12320.ckpt',#None,
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    
    args = parser.parse_args(src_args)
    return args
    

if __name__ == "__main__":
    main()