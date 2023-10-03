import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as M
from torchvision.transforms import transforms as T
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, BinaryAccuracy

import pytorch_lightning as pl
import kornia as K
from PIL import Image
import pandas as pd
from os.path import join

import albumentations as A
import cv2
import numpy as np
import glob 
import os
import random
import math


class ModelBuilder:

    ARCHS = {
        "eff": M.efficientnet_v2_s,
        "res": M.wide_resnet50_2,
        "swin": M.swin_v2_b,
        "vit_l_16": M.vit_l_16,
        "eff_l": M.efficientnet_v2_l,
    }
    WEIGHTS = {
        "eff": M.EfficientNet_V2_S_Weights.DEFAULT,
        "res": M.Wide_ResNet50_2_Weights.DEFAULT,
        "swin": M.Swin_V2_B_Weights.DEFAULT,
        "vit_l_16": M.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        "eff_l": M.EfficientNet_V2_L_Weights.DEFAULT,
    }

    @classmethod
    def build(cls, arch, num_classes, trained=True) -> nn.Module:
        assert arch in cls.ARCHS
        weights = cls.WEIGHTS[arch] if trained else None
        model = cls.ARCHS[arch](weights=weights)
        return eval(f"cls.{arch.split('_')[0]}")(model, num_classes)

    @staticmethod
    def eff(model, NUM_CLASSES):
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=in_feat, out_features=NUM_CLASSES),
        )
        return model

    @staticmethod
    def res(model, NUM_CLASSES):
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_features=in_feat, out_features=NUM_CLASSES)
        return model

    @staticmethod
    def swin(model, NUM_CLASSES):
        in_feat = model.head.in_features
        model.head = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_feat, out_features=NUM_CLASSES),
        )
        return model
    
    @staticmethod
    def vit(model, NUM_CLASSES):
        in_feat = model.heads[-1].in_features
        model.heads = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=in_feat, out_features=NUM_CLASSES),
        )
        return model

def image_to_tensor_train(path):
    tensor = Image.open(path)
    if tensor.size[1] > tensor.size[0] and torch.rand(1) > 0.5:
        tensor = tensor.rotate(90, expand=True)
    return T.ToTensor()(tensor.convert("RGB")).float()  

def image_to_tensor(path):
    tensor = Image.open(path)
    return T.ToTensor()(tensor.convert("RGB")).float()   


class ImageDataset(Dataset):
    """Torch Dataset for image loading and preprocessing with sample generation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        transforms: nn.Sequential = None,
        root: str = "",
        train: bool = True,
        group: str = 'sex_position',
    ):
        """
        :param data: pandas DataFrame
        :param transforms: image transforms that will be invoked in __getitem__
        :param root: path to dir that must contain subdirs 'background', 'male', 'female', defaults to ""
        :param train: whether train or not, defaults to True
        :param group: group of categories that affects the way of image generation, defaults to 'sex_position'
        """
        
        self.data = data
        self.root = root
        self.group = group
        self.train = train
        self.transforms = transforms
        
        available_groups = ['sex_position', 'tits_size']
        assert self.group in available_groups
        
        self.num_classes = len(data.columns) - 1
        self.num2label = {i: col for i, col in enumerate(data.columns[1:])}
        

    @torch.no_grad()
    def __getitem__(self, idx):
        
        # Get random background img from "{root}/background/"
        bg_img = self.get_bg_img()
        
        # Get foreground img by index using dirs "{root}/male/" and "{root}/female/"
        fg_img = self.get_fg_img(idx, (bg_img.shape[1], bg_img.shape[0]))
        
        img = self.merge_images(bg_img, fg_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        if self.train and torch.rand(1) > 0.7:
            img = self.resize_tensor_with_pad(img, (320, 240))
        else:
            img = self.resize_tensor_with_pad(img, (640, 480))
        
        img = img.squeeze(0).to(torch.float16)
        label = torch.tensor(self.data.iloc[idx][1:], dtype=torch.float16)
        return img, label

    def __len__(self):
        return len(self.data)
    
    def resize_tensor_with_pad(self, img: torch.Tensor, size: tuple) -> torch.Tensor:
        w, h = size
        if img.size()[1] < img.size()[2]:                    
            img = K.augmentation.LongestMaxSize(w)(img)
        else:
            img = K.augmentation.LongestMaxSize(h)(img)
        img = K.augmentation.PadTo((h, w), keepdim=True)(img)
        return img

    def resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Resize img with pad keeping original ratio"""
        w, h = size
        img = A.LongestMaxSize(max_size=min(w, h))(image=img)['image']
        img = A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT)(image=img)['image']
        return img
    
    def get_bg_img(self) -> np.ndarray:
        if self.group == 'tits_size':
            img = self.get_bg_tits_size()
        else: # default - 'sex_position'
            img = self.get_bg_sex_position()
        return img
    
    def get_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        if self.group == 'tits_size':
            img = self.get_fg_tits_size(idx, bg_size)
        else: # default - 'sex_position'
            img = self.get_fg_sex_position(idx, bg_size)
        return img
    
    def get_data_by_idx(self, idx) -> tuple:
        """Parse info from self.data by index

        :param idx: current index
        :return: tuple of image name and class number 
        """
        data = self.data.iloc[idx]
        cur_class = data[1:].values.argmax()
        img_fn = data['path']
        img_name = os.path.splitext(img_fn)[0]
        return img_name, cur_class
    
    def get_fg_sex_position(self, idx, bg_size):
        """Create foreground for 'sex position' group. 
        Find all male and female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_name, cur_class = self.get_data_by_idx(idx)
        
        fg_img_paths = glob.glob(os.path.join(self.root, 'male', f"{img_name}_*"))
        fg_img_paths += glob.glob(os.path.join(self.root, 'female', f"{img_name}_*"))
        fg_imgs = [cv2.imread(path) for path in fg_img_paths]
        
        width, height = bg_size
        res_img = np.zeros((height, width, 3), dtype='uint8')
        for fg_img in fg_imgs:
            fg_img = self.resize_with_pad(fg_img, (width, height))
            res_img += fg_img
        
        # Spooning pose always horizontal so it doesnt need to be rotated
        degrees = 0 if self.num2label[cur_class] == 'spooning' else 70
        res_img = self.warp_img(res_img, degrees=degrees)
        return res_img

    
    def get_fg_tits_size(self, idx, bg_size):
        """Create foreground for 'tits_size' group. 
        Find only female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_name, cur_class = self.get_data_by_idx(idx)
        
        fg_img_paths = glob.glob(os.path.join(self.root, 'female', f"{img_name}_*"))
        fg_imgs = [cv2.imread(path) for path in fg_img_paths]
        
        width, height = bg_size
        res_img = np.zeros((height, width, 3), dtype='uint8')
        for fg_img in fg_imgs:
            fg_img = self.resize_with_pad(fg_img, (width, height))
            res_img += fg_img
        
        res_img = self.warp_img(res_img)
        return res_img
    
    def read_random_bg_img(self) -> np.ndarray:
        """Read random background image listed in self.data"""
        idx = random.randint(0, len(self.data) - 1)
        data = self.data.iloc[idx]
        img_fn = data['path']
        
        # Note: I dont know how to handle .gif files corectly (cv2.imread cant read this), 
        # that's why I just make black image instead 
        if os.path.splitext(img_fn)[1] == '.gif': 
            img = np.zeros((640, 640, 3), dtype='uint8')
        else:
            img = cv2.imread(os.path.join(self.root, 'background', img_fn))
        return img
    
    def get_bg_sex_position(self):
        img = self.read_random_bg_img()
        return img
    
    def get_bg_tits_size(self, p=0.5):
        """Read random background image and add a random male with probability 'p' """
        img = self.read_random_bg_img()
        
        if random.random() < p:
            male_paths = glob.glob(os.path.join(self.root, 'male', f"*"))
            male_path = random.choice(male_paths)
            
            male_img = cv2.imread(male_path)
            male_img = self.resize_with_pad(male_img, (img.shape[1], img.shape[0]))
            male_img = self.warp_img(male_img)
            img = self.merge_images(img, male_img)    
        
        return img
    
    
    def warp_img(self, 
                 img: np.ndarray, 
                 degrees=20, 
                 translate=0.1, 
                 scale=0.3,
                 shear=30,
                 perspective=0.0):
        
        mask = self.get_black_mask(img)
        
        # NOTE: I am not sure that this is works correctly, so let it be commented at the moment
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if len(contours) > 0:
        #     common_cnt = np.concatenate(contours, axis=0)
        #     x, y, w, h = cv2.boundingRect(common_cnt)
        #     x2, y2 = x + w, y + h
        #     height, width = img.shape[:2]
        #     translate = min(x / width, (width - x2) / width, y / height, (height - y2) / height)
        # else:
        #     translate = 0.1
        
        warp_mat = self.get_random_perspective_transform(img.shape, 
                                                         translate=translate, 
                                                         degrees=degrees, 
                                                         scale=scale,
                                                         shear=shear)
        img = cv2.warpPerspective(img, warp_mat, (img.shape[1], img.shape[0]))
        return img
        
    def merge_images(self, background_img: np.ndarray, foreground_img: np.ndarray):
        mask = self.get_black_mask(foreground_img)
        mask_inv = cv2.bitwise_not(mask)
        
        bg_without_fg = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
        final_img = cv2.add(bg_without_fg, foreground_img)
        
        return final_img

    def get_black_mask(self, img: np.ndarray, threshold: int = 5):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def get_resizing_transform(self, width: int, height: int):
        transforms = [
            A.LongestMaxSize(max_size=min(width, height)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
        ]
        return A.Compose(transforms)
    
    def get_random_perspective_transform(self,
                                         img_shape: tuple,
                                         degrees=20,
                                         translate=0.4,
                                         scale=0.2,
                                         shear=30,
                                         perspective=0.0) -> np.ndarray:
        height = img_shape[0]
        width = img_shape[1]

        # Center
        C = np.eye(3)
        C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        # s = random.uniform(scale - scale_range, scale + scale_range)
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        return M


class InferDataset(Dataset):
    """
    Description:
        Torch DataSet for inference.
        :data: list
        :transforms: nn.Sequential
    """

    def __init__(
        self,
        data: list,
        transforms: nn.Sequential,
    ):
        self.data = data
        self.transforms = transforms

    def parse_data(self, data):
        img = image_to_tensor(data)
        img_id = data.split("/")[-1]
        return img, img_id.split(".")[0]

    @torch.no_grad()
    def __getitem__(self, idx):
        data = self.data[idx]
        img, img_id = self.parse_data(data)
        if self.transforms:
            img = self.transforms(img)
        return img.to(torch.float16), img_id

    def __len__(self):
        return len(self.data)


class EfficientLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num2label: dict = None,
        batch_size: int = 24,
        decay: float = 0.2,
        augmentation: nn.Sequential = None,
        weights: list = None,
    ):
        super().__init__()

        self.num2label = num2label

        self.model = model
        self.model.num2label = num2label
        self.transform = augmentation()

        self.batch_size = batch_size
        self.decay = decay
        self.lr = 1e-4
        self.sigmoid = nn.Sigmoid()
        self.cross_entropy = nn.BCEWithLogitsLoss(
            # pos_weight=self.get_pos_weight(weights),
            reduction="none",
        )
        self.build_metrics()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        # if self.trainer.training:
        x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        # => we perform GPU/Batched data augmentation
        logits = self.forward(x)
        loss = self.cross_entropy(logits, y)
        #################################
        pt = torch.exp(-loss)
        loss = (2 * (1 - pt) ** 2 * loss).sum()
        #################################
        log = self.get_log("train", self.sigmoid(logits), y)
        log["train_loss"] = loss
        self.log_dict(
            log,
            on_step=True,
            logger=True,
            prog_bar=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        # => we perform GPU/Batched data augmentation
        logits = self.forward(x)
        loss = self.cross_entropy(logits, y)
        #################################
        pt = torch.exp(-loss)
        loss = (2 * (1 - pt) ** 2 * loss).sum()
        #################################
        log = self.get_log("val", self.sigmoid(logits), y)
        log["val_loss"] = loss
        self.log_dict(
            log,
            on_step=True,
            logger=True,
            prog_bar=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return loss

    def get_pos_weight(self, weights):
        pos_weight = torch.tensor(weights) * 3 if type(weights) == list else None
        return pos_weight

    def get_log(self, mod, logits, y):
        acc_m, f1_m, f1_M, f1_N = self.get_metrics(logits, y)
        if len(self.num2label) == 1:
            return {mod + "_acc_micro": acc_m}
        log = {
            mod + "_acc_micro": acc_m,
            mod + "_F1_micro": f1_m,
            mod + "_F1_Macro": f1_M,
        }
        for key, value in self.num2label.items():
            log[mod + f"_f1_{value}"] = f1_N[int(key)]

        return log

    def build_metrics(self):
        num_labels = len(self.num2label) 
        if num_labels == 1:
            self.accuracy1 = BinaryAccuracy()
            self.F1_M = lambda x, y: torch.tensor(0)
            self.F1_m = lambda x, y: torch.tensor(0)
            self.F1_N = lambda x, y: torch.tensor(0)
        else:
            self.accuracy1 = MultilabelAccuracy(
                num_labels=num_labels,
                average="micro",
            )
            self.F1_M = MultilabelF1Score(
                num_labels=num_labels,
                average="macro",
            )
            self.F1_m = MultilabelF1Score(
                num_labels=num_labels,
                average="micro",
            )
            self.F1_N = MultilabelF1Score(
                num_labels=num_labels,
                average="none",
            )

    @torch.no_grad()
    def get_metrics(self, logits, y):
        acc_m = self.accuracy1(logits, y)
        f1_m = self.F1_m(logits, y)
        f1_M = self.F1_M(logits, y)
        f1_N = self.F1_N(logits, y)
        return acc_m, f1_m, f1_M, f1_N

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.decay, eps=1e-7
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode="min", factor=0.5, patience=10
            ),
            "monitor": "train_loss",
            "name": "LR",
        }
        return [optimizer], [scheduler]
