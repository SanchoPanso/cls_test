import cv2
import numpy as np
import glob 
import os
import random
import math
import albumentations as A

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
from typing import List


class ModelBuilder:

    ARCHS = {
        "eff": M.efficientnet_v2_s,
        "eff_softmax": M.efficientnet_v2_s,
        "res": M.wide_resnet50_2,
        "swin": M.swin_v2_b,
        "vit_l_16": M.vit_l_16,
        "eff_l": M.efficientnet_v2_l,
    }
    WEIGHTS = {
        "eff": M.EfficientNet_V2_S_Weights.DEFAULT,
        "eff_softmax": M.EfficientNet_V2_S_Weights.DEFAULT,
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
    def eff_softmax(model, NUM_CLASSES):
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=in_feat, out_features=NUM_CLASSES),
            nn.Softmax(dim=1),
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
        self.transform = augmentation() if augmentation else None

        self.batch_size = batch_size
        self.decay = decay
        self.lr = 1e-4
        self.sigmoid = nn.Sigmoid()
        self.cross_entropy = nn.BCEWithLogitsLoss(
            # pos_weight=self.get_pos_weight(weights),
            reduction="none",
        )
        self.build_metrics()
        self.save_hyperparameters(ignore=['model'])

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
        #log = self.get_log("train", self.sigmoid(logits), y)
        log = self.get_log("train", logits, y)
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
        #log = self.get_log("val", self.sigmoid(logits), y)
        log = self.get_log("val", logits, y)
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
