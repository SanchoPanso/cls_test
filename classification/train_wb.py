import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from train.augmentation import DataAugmentation, PreProcess
from train.model import EfficientLightning, ImageDataset
from train.service import TrainWrapper
import argparse

torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Type path to: model, json, data(optional)"
    )
    parser.add_argument(
        "--cat", dest="cat", type=str, default="tits", help="Category", required=False
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
        default="val",
        help="Callback mode",
        required=False,
    )
    parser.add_argument(
        "--decay",
        dest="decay",
        type=float,
        default=0.2,
        help="Callback mode",
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
    args = parser.parse_args()

    WRAPPER = TrainWrapper(
        cfg=args,
        imageDataset=ImageDataset,
        preProc=PreProcess,
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
        max_epochs=60,
        precision=16,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=WRAPPER.get_callbacks(),
    )
    # Train the model ⚡
    trainer.fit(model, WRAPPER.train_loader, WRAPPER.val_loader)

    path_ = WRAPPER.get_best_model(model)
