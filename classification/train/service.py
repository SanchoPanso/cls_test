from datetime import timedelta
import os
import sys
import json
import glob
from pathlib import Path
from io import StringIO
from typing import Optional
from lightning_fabric.utilities.types import _PATH
import pytorch_lightning as pl

import torch
import numpy as np
import pandas as pd
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .model import ModelBuilder
from .datasets import GenerativeDataset, get_dataset_by_group
from .augmentation import PreProcess, DataAugmentation

sys.path.append(str(Path(__file__).parent.parent))
from data_info import count_classes
from utils.logger import get_logger

LOGGER = get_logger(__name__)
seed_everything(1)


class TrainWrapper:
    """
    Attrs
        model: torch nn.Module
        train_loader: torch DataLoader (train)
        val_loader: torch DataLoader (val)
        weights: weights for each class in dataset
        num2label: match num to label
        label2num: match label to num
        save_dir: path for store comet ml
        experiment_name: name of the current train
        cat: train category
        arch: architecture

    """

    def __init__(
            self, cfg, num_workers=32, pre_train=True
    ) -> None:
        self.cat = cfg.cat
        self.bg_cat = cfg.bg_cat
        self.mode = cfg.mode
        self.arch = cfg.arch
        self.decay = cfg.decay
        self.gray = cfg.gray
        self.vflip = cfg.vflip
        self.batch_size = cfg.batch_size

        self.dataset_root_path = cfg.data_path
        self.dataset_meta_path = os.path.join(cfg.data_path, cfg.datasets_dir, cfg.cat + '.json')
        self.background_meta_path = os.path.join(cfg.data_path, cfg.datasets_dir, cfg.bg_cat + '.json')
        self.masks_subdir = cfg.segments_dir #cfg.masks_dir
        self.pictures_subdir = cfg.images_dir
        self.badlist_path = cfg.badlist_path

        self.models_dir = os.path.join(cfg.data_path, cfg.models_dir)
        self.num_workers = num_workers
        self.__get_class_decoder()
        self.__get_dataloader()
        self.__save_log_dir()

        self.model = ModelBuilder.build(
            arch=cfg.arch, num_classes=self.num_classes, trained=pre_train
        )

    def __get_class_decoder(self):
        with open(self.dataset_meta_path) as f:
            json_ = json.load(f)
        self.cats = json_["cat"]
        self.num2label = json_["num2label"]

        self.num_classes = len(self.num2label)
        self.weights = json_["weights"]  # TODO: check this
        self.train_pd = pd.read_json(StringIO(json_["data"])).reset_index(drop=True)
        LOGGER.info(f"Class names: {self.num2label}")
        LOGGER.info(f"Class index of maximum weight: {self.weights.index(max(self.weights))}")
        
        if self.mode == "train":
            self.train_pd, self.val_pd = train_test_split(
                self.train_pd,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=self.train_pd[
                    self.num2label[str(self.weights.index(max(self.weights)))]
                ],
            )
            self.train_pd = self.train_pd.reset_index(drop=True)
            self.val_pd = self.val_pd.reset_index(drop=True)
        
        elif self.mode == "all":
            #self.train_pd = self.train_pd.reset_index(drop=True)
            self.val_pd = shuffle(self.train_pd).reset_index(drop=True)

    def __get_dataloader(self):
        with open(self.background_meta_path) as f:
            bg_data = pd.read_json(StringIO(json.load(f)['data']))

        LOGGER.info('Prepare Train Subset')
        self.train_set = get_dataset_by_group(
            group=self.cat,
            foreground_data=self.train_pd,
            background_data=bg_data,
            masks_dir=os.path.join(self.dataset_root_path, self.masks_subdir),
            pictures_dir=os.path.join(self.dataset_root_path, self.pictures_subdir),
            transforms=PreProcess(gray=self.gray, vflip=self.vflip, arch=self.arch),
            train=True,
            badlist_path=self.badlist_path,
        )
        
        LOGGER.info('Prepare Val Subset')
        self.val_set = get_dataset_by_group(
            group=self.cat,
            foreground_data=self.val_pd,
            background_data=bg_data,
            masks_dir=os.path.join(self.dataset_root_path, self.masks_subdir),
            pictures_dir=os.path.join(self.dataset_root_path, self.pictures_subdir),
            transforms=PreProcess(gray=False, vflip=False, arch=self.arch),
            train=False,
            badlist_path=self.badlist_path,
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            pin_memory=True,
            sampler=self.get_sampler(),
            # shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def __save_log_dir(self):
        log_dir = os.path.join(self.models_dir, self.cat)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = self.models_dir
        self.experiment_name = f"v__{str(len(os.listdir(log_dir)))}_{self.mode}_{self.arch}_{self.batch_size}_{self.decay}"

    # @staticmethod
    def get_callbacks(self, monitor="val_F1_Macro_epoch", mode="max"):
        if len(self.num2label) <= 1:
            monitor = "val_acc_micro_epoch"
            mode = "max"
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=monitor, mode=mode, save_weights_only=True
        )
        torchscript_callback = CustomModelCheckpoint(
            dirpath=os.path.join(self.save_dir, self.cat, self.experiment_name, 'torchscripts'),
            save_top_k=1, 
            monitor=monitor, 
            mode=mode, 
            save_weights_only=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, torchscript_callback, lr_monitor]

    def get_sampler(self):
        train_pd = self.train_set.foreground_data
        cols = train_pd.columns[1:]
        df_len = len(train_pd)
        my_weights = torch.tensor([0.] * df_len)
        all_ = 0
        for col in cols:
            ret = train_pd[col][train_pd[col] == 1].sum()
            indexes_of_cls = np.where((train_pd[col] == 1).values)[0].tolist()
            my_weights[indexes_of_cls] = 1 / len(indexes_of_cls)
            all_ += ret
        my_weights[my_weights == 0] = 1 / (df_len - all_)
        return WeightedRandomSampler(my_weights, num_samples=df_len, replacement=True)

    def get_best_model(self, model):
        path_ = glob.glob(
            os.path.join(self.save_dir, self.cat, self.experiment_name, "checkpoints/*.ckpt")
        )
        for path in path_:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            script = torch.jit.script(model.model)
            # save for use in production environment
            extra_files = json.dumps(self.num2label)
            data_info = count_classes(self.train_set.foreground_data).to_json()
            torch.jit.save(
                script,
                path.replace("ckpt", "pt"),
                _extra_files={
                    "num2label.txt": extra_files,
                    "data_info.txt": data_info,
                },
            )
        return path_


def get_class_decoder(cat, source):
    # need source for every category!
    with open(os.path.join(source, f"{cat}.json")) as f:
        json_ = json.load(f)

    num2label = json_["num2label"]
    # __train_pd, __val_pd = pd.read_json(json_["train"]), pd.read_json(json_["val"])
    # data = pd.read_json(json_["data"])
    weights = json_["weights"]
    print(weights)
    return num2label, weights


class CustomModelCheckpoint(ModelCheckpoint):
    FILE_EXTENSION = '.pt'
    
    def __init__(
        self,
        dirpath: _PATH | None = None, 
        filename: str | None = None, 
        monitor: str | None = None, 
        verbose: bool = False, 
        save_last: bool | None = None, 
        save_top_k: int = 1, 
        save_weights_only: bool = False, 
        mode: str = "min", 
        auto_insert_metric_name: bool = True, 
        every_n_train_steps: int | None = None, 
        train_time_interval: timedelta | None = None, 
        every_n_epochs: int | None = None, 
        save_on_train_epoch_end: bool | None = None):
        
        super().__init__(
            dirpath, 
            filename, 
            monitor, 
            verbose, 
            save_last, 
            save_top_k, 
            save_weights_only, 
            mode, 
            auto_insert_metric_name, 
            every_n_train_steps, 
            train_time_interval, 
            every_n_epochs, 
            save_on_train_epoch_end
        )
        
    
    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        pl_model = trainer.model
        script = torch.jit.script(pl_model.model)
        extra_files = json.dumps(pl_model.num2label)
        #data_info = count_classes(self.train_set.foreground_data).to_json()
        os.makedirs(self.dirpath, exist_ok=True)
        
        torch.jit.save(
            script,
            filepath,
            _extra_files={
                "num2label.txt": extra_files,
                #"data_info.txt": data_info,
            },
        )
        
        #return super()._save_checkpoint(trainer, filepath)

