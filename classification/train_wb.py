import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import logging
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from typing import Sequence

sys.path.append(str(Path(__file__).parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from train.service import TrainWrapper
from utils.cfg_handler import get_cfg
from utils.utils import dict2str
from utils.logger import get_logger

LOGGER = get_logger(os.path.splitext(os.path.basename(__file__))[0])


def main():
    torch.cuda.empty_cache()

    args = parse_args()
    cfg = get_cfg(args.cfg)
    cfg.update(vars(args))
    LOGGER.info(f'Configuration: {dict2str(cfg)}')

    WRAPPER = TrainWrapper(
        cfg=cfg,
        num_workers=cfg.num_workers,
    )
    
    LOGGER.info(f'Result will be saved in ' 
                f'\033[1m{os.path.join(WRAPPER.save_dir, WRAPPER.cat, WRAPPER.experiment_name)}\033[0m')
        
    model = EfficientLightning(
        model=WRAPPER.model,
        num2label=WRAPPER.num2label,
        batch_size=WRAPPER.batch_size,
        decay=WRAPPER.decay,
        augmentation=DataAugmentation(),
        weights=WRAPPER.weights,
    )
    
    if cfg.pretrained:
        cfg.pretrained = os.path.join(cfg.data_path, cfg.models_dir, cfg.pretrained)
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
    
    csv_logger = CSVLogger(
        save_dir=os.path.join(WRAPPER.save_dir, WRAPPER.cat, WRAPPER.experiment_name),
        name='lightning_logs',
    )

    # Initialize a trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=[cfg.gpu],
        max_epochs=cfg.epochs, # 60,
        precision=16,
        log_every_n_steps=1,
        logger=[wandb_logger, csv_logger],
        callbacks=WRAPPER.get_callbacks(),
    )

    # Train the model ⚡
    create_train_examples(WRAPPER)
    trainer.fit(model, WRAPPER.train_loader, WRAPPER.val_loader)


def create_train_examples(wrapper: TrainWrapper, num_of_batches=3):
    train_batches_dir = os.path.join(wrapper.save_dir, wrapper.cat, wrapper.experiment_name, 'train_batches')
    os.makedirs(train_batches_dir, exist_ok=True)
    
    for i, batch in enumerate(wrapper.train_loader):
        if i >= num_of_batches:
            break
        
        imgs, labels = batch
        batch_size = imgs.shape[0]
        height = int(np.ceil(np.sqrt(batch_size)))

        plt.figure(figsize=(45, 15))
        for j in range(batch_size):
            img = imgs[j].permute(1, 2, 0).cpu().float().numpy()
            label = labels[j].cpu().numpy().argmax()
            name = wrapper.num2label[str(label)]
            
            plt.subplot(height, height, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(name)
            plt.imshow(img)
            
        plt.savefig(os.path.join(train_batches_dir, f'train_batch_{i}.jpg'))     


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
        default="hair_type", 
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
    
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=32)
    
    args = parser.parse_args(src_args)
    return args
    

if __name__ == "__main__":
    main()