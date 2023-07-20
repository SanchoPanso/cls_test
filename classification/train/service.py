from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from .model import ModelBuilder
import pandas as pd
import os
from os.path import join
from glob import glob
import json
from lightning_fabric.utilities.seed import seed_everything

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
    SOURCE = "/home/timssh/ML/TAGGING/DATA/datasets"
    DATA = "/home/timssh/ML/TAGGING/DATA/picture"
    MODEL = "/home/timssh/ML/TAGGING/DATA/models"

    def __init__(
        self, cfg, imageDataset, preProc, num_workers=32, pre_train=True
    ) -> None:
        self.cat = cfg.cat
        self.mode = cfg.mode
        self.arch = cfg.arch
        self.decay = cfg.decay
        self.gray = cfg.gray
        self.vflip = cfg.vflip
        self.batch_size = cfg.batch_size
        self.num_workers = num_workers
        self.__imageDataset = imageDataset
        self.__preProc = preProc
        self.__get_class_decoder()
        self.__get_dataloader()
        self.__save_log_dir()
        self.model = ModelBuilder.build(
            arch=cfg.arch, num_classes=self.num_classes, trained=pre_train
        )

    def __get_class_decoder(self):
        with open(join(self.SOURCE, f"{self.cat}.json")) as f:
            json_ = json.load(f)
        self.cats = json_["cat"]
        self.num2label = json_["num2label"]
        self.num_classes = len(json_["weights"])
        self.weights = json_["weights"]
        self.__train_pd = pd.read_json(json_["data"])
        print(self.num2label, self.weights.index(max(self.weights)))
        if self.mode == "train":
            self.__train_pd, self.__val_pd = train_test_split(
                self.__train_pd,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=self.__train_pd[
                    self.num2label[str(self.weights.index(max(self.weights)))]
                ],
            )
            self.__train_pd = self.__train_pd.reset_index(drop=True)
            self.__val_pd = self.__val_pd.reset_index(drop=True)
        if self.mode == "all":
            self.__val_pd = shuffle(self.__train_pd)

    def __get_dataloader(self):
        # REQUIRED
        train_set = self.__imageDataset(
            self.__train_pd,
            self.__preProc(gray=self.gray, vflip=self.vflip, arch=self.arch),
            train=True,
            root=self.DATA,
        )
        val_set = self.__imageDataset(
            self.__val_pd,
            self.__preProc(gray=False, vflip=False, arch=self.arch),
            train=False,
            root=self.DATA,
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def __save_log_dir(self):
        log_dir = join(self.MODEL, self.cat)
        os.makedirs(log_dir, exist_ok=True)
        self.save_dir = self.MODEL
        self.experiment_name = f"version_{str(len(os.listdir(log_dir)))}_{self.mode}_{self.arch}_{self.batch_size}_{self.decay}"

    # @staticmethod
    def get_callbacks(self, monitor="val_F1_Macro_epoch", mode="max"):
        if len(self.num2label) <= 1:
            monitor = "val_loss_epoch"
            mode = "min"
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=monitor, mode=mode, save_weights_only=True
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, lr_monitor]

    def get_best_model(self, model):
        path_ = glob(
            join(self.save_dir, self.cat, self.experiment_name, "checkpoints/*.ckpt")
        )
        for path in path_:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            script = torch.jit.script(model.model)
            # save for use in production environment
            torch.jit.save(script, path.replace("ckpt", "pt"))
        return path_


def get_class_decoder(cat, source):
    # need source for every category!
    with open(join(source, f"{cat}.json")) as f:
        json_ = json.load(f)

    num2label = json_["num2label"]
    # __train_pd, __val_pd = pd.read_json(json_["train"]), pd.read_json(json_["val"])
    # data = pd.read_json(json_["data"])
    weights = json_["weights"]
    print(weights)
    return num2label, weights
