from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from .model import ModelBuilder
from .datasets import ImageDataset, get_dataset_by_group
from .augmentation import PreProcess, DataAugmentation
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

    def __init__(
        self, cfg, num_workers=32, pre_train=True
    ) -> None:
        self.cat = cfg.cat
        self.mode = cfg.mode
        self.arch = cfg.arch
        self.decay = cfg.decay
        self.gray = cfg.gray
        self.vflip = cfg.vflip
        self.batch_size = cfg.batch_size
        self.dataset_meta_path = os.path.join(cfg.data_path, cfg.dataset_path)
        self.dataset_root_path = os.path.join(cfg.data_path, cfg.masks_dir)
        self.models_dir = os.path.join(cfg.data_path, cfg.models_dir)
        #self.DATA = self.ROOT + "masks" if cfg.masks else self.ROOT + "picture"
        self.num_workers = num_workers
        # self.__imageDataset = imageDataset
        # self.__preProc = preProc
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
        train_set = get_dataset_by_group(
            group='tits_size',
            data=self.__train_pd,
            transforms=PreProcess(gray=self.gray, vflip=self.vflip, arch=self.arch),
            train=True,
            root=self.dataset_root_path,
        )
        val_set = get_dataset_by_group(
            group='tits_size',
            data=self.__val_pd,
            transforms=PreProcess(gray=False, vflip=False, arch=self.arch),
            train=False,
            root=self.dataset_root_path,
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            pin_memory=True,
            sampler=self.get_sampler(),
            # shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def __save_log_dir(self):
        log_dir = join(self.models_dir, self.cat)
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
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, lr_monitor]

    def get_sampler(self):
        cols = self.__train_pd.columns[1:]
        df_len = len(self.__train_pd)
        my_weights = torch.tensor([0.] * df_len)
        all_ = 0
        for col in cols:
            ret = self.__train_pd[col][self.__train_pd[col]==1].sum()
            indexes_of_cls = list(self.__train_pd[col][self.__train_pd[col]==1].index)
            my_weights[indexes_of_cls] = 1 / len(indexes_of_cls)
            all_ += ret
        my_weights[my_weights == 0] = 1 / (df_len - all_)
        return WeightedRandomSampler(my_weights, num_samples=df_len, replacement=True)

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
            extra_files = json.dumps(self.num2label)
            torch.jit.save(
                script,
                path.replace("ckpt", "pt"),
                _extra_files={"num2label.txt": extra_files},
            )
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
