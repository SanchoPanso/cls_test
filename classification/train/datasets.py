import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as M
from torchvision.transforms import transforms as T
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, BinaryAccuracy

import pytorch_lightning as pl
from PIL import Image
import pandas as pd
from os.path import join
from typing import List, Dict

import albumentations as A
import cv2
import numpy as np
import glob 
import os
import random
import math


class InferenceDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        images_dirs: List[str] | str,
        transforms: nn.Sequential = None,):
    
        self.data = data
        self.images_dirs = [images_dirs] if type(images_dirs) == str else images_dirs
        self.transforms = transforms
        
        data_paths = data['path'].tolist()
        data_names = set(map(lambda x: os.path.splitext(x)[0], data_paths))
        
        self.image_paths = []
        
        for imd in images_dirs:
            all_image_paths = os.listdir(imd)
            
            for fn in all_image_paths:
                name = '_'.join(fn.split('_')[:-1])
                if name in data_names:
                    path = os.path.join(imd, fn)
                    self.image_paths.append(path)
        
        
    @torch.no_grad()
    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_with_pad(img, (640, 480))
        img = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = img.squeeze(0).to(torch.float16)
        return img, img_path

    def __len__(self):
        return len(self.image_paths)
    
    def resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Resize img with pad keeping original ratio"""
        w, h = size
        img = A.LongestMaxSize(max_size=min(w, h))(image=img)['image']
        img = A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT)(image=img)['image']
        return img    
    

class TrainDataset(Dataset):
    """Dataset for image loading and preprocessing with sample generation.
    This class implements default behaviour of image generation, that can be 
    extended by inherited classes for specific categories.
    """
    
    train: bool
    transforms: nn.Sequential
    num_classes: int
    num2label: Dict[int, str]

    def __init__(
        self,
        foreground_data: pd.DataFrame,
        background_data: pd.DataFrame,
        masks_dir: str,
        pictures_dir: str,
        transforms: nn.Sequential = None,
        train: bool = True,
        badlist_path: str = None,
    ):
        """
        :param data: pandas DataFrame
        :param transforms: image transforms that will be invoked in __getitem__
        :param root: path to dir, defaults to ""
        :param train: whether train or not, defaults to True
        :param group: group of categories that affects the way of image generation, defaults to 'sex_position'
        """
        
        if badlist_path:
            with open(badlist_path) as f:
                text = f.read().strip()
                badlist = [] if text == '' else text.split('\n')
            self.foreground_data = self.delete_badlist(foreground_data, badlist)
        else:
            self.foreground_data = foreground_data
        
        self.background_data = background_data
        self.masks_dir = masks_dir
        self.pictures_dir = pictures_dir
        self.train = train
        self.transforms = transforms
        
        self.num_classes = len(foreground_data.columns) - 1
        self.num2label = {i: col for i, col in enumerate(foreground_data.columns[1:])}

    @torch.no_grad()
    def __getitem__(self, idx):
        
        bg_img = self.get_bg_img(idx)
        bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
        fg_img = self.get_fg_img(idx, (bg_w, bg_h))
        img = self.merge_images(bg_img, fg_img)
        
        # Random decrease of th resolution for training
        if self.train and torch.rand(1) > 0.7:
            img = self.resize_with_pad(img, (320, 240))
        img = self.resize_with_pad(img, (640, 480))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = img.squeeze(0).to(torch.float16)
        label = torch.tensor(self.foreground_data.iloc[idx].tolist()[1:], dtype=torch.float16)
        return img, label

    def __len__(self):
        return len(self.foreground_data)
    
    @classmethod
    def delete_badlist(cls, df: pd.DataFrame, badlist: List[str]) -> pd.DataFrame:
        
        df.index = df[df.columns[0]]        
        clear_badlist = []
        for b in badlist:
            if b in df.index:
                clear_badlist.append(b)

        df_drop = df.drop(clear_badlist)
        df.index = range(len(df))
        df_drop.index = range(len(df_drop))
        return df
        
    def resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Resize img with pad keeping original ratio"""
        w, h = size
        h0, w0 = img.shape[:2]
        if w / w0 < h / h0:
            img = A.Resize(w * h0 // w0, w)(image=img)['image']
        else:
            img = A.Resize(h, h * w0 // h0)(image=img)['image']
        img = A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT)(image=img)['image']
        return img
    
    def get_bg_img(self, idx: int) -> np.ndarray:
        img = self.read_random_bg_img()
        img = self.augment_bg_img(img)
        return img
    
    def get_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        """Create foreground for 'sex position' group. 
        Find all male and female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_name, cur_class = self.get_data_by_idx(idx)
        
        fg_img_paths = glob.glob(os.path.join(self.masks_dir, 'male', f"{img_name}_*"))
        fg_img_paths += glob.glob(os.path.join(self.masks_dir, 'female', f"{img_name}_*"))
        fg_imgs = [cv2.imread(path) for path in fg_img_paths]
        
        width, height = bg_size
        res_img = np.zeros((height, width, 3), dtype='uint8')
        
        for fg_img in fg_imgs:
            fg_img = self.resize_tensor_with_pad(fg_img, (width, height))
            res_img += fg_img
        
        res_img = self.warp_img(res_img)
        return res_img
    
    def get_data_by_idx(self, idx) -> tuple:
        """Parse info from self.data by index

        :param idx: current index
        :return: tuple of image name and class number 
        """
        data = self.foreground_data.iloc[idx]
        if data[1:].values.sum() == 0:
            cur_class = -1
        else:
            cur_class = data[1:].values.argmax()
        
        img_fn = data['path']
        img_name = os.path.splitext(img_fn)[0]
        return img_name, cur_class
    
    def read_random_bg_img(self) -> np.ndarray:
        """Read random background image listed in self.data"""
        idx = random.randint(0, len(self.background_data) - 1)
        data = self.background_data.iloc[idx]
        img_fn = data['path']
        
        # Note: I dont know how to handle .gif files corectly (cv2.imread cant read this), 
        # that's why I just make black image instead 
        if os.path.splitext(img_fn)[1] == '.gif': 
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.imread(os.path.join(self.pictures_dir, img_fn))
        return img
    
    def augment_bg_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[0:2]
        crop_h, crop_w = max(1, h // 2), max(1, w // 2)
        
        img = A.RandomResizedCrop(crop_h, crop_w, p=0.5)(image=img)['image']
        img = A.RandomBrightnessContrast(p=0.5)(image=img)['image']
        img = A.GaussNoise(p=0.2)(image=img)['image']
        img = A.HorizontalFlip(p=0.5)(image=img)['image']

        return img
    
    def warp_img(self, img: np.ndarray) -> np.ndarray:
        img = A.ShiftScaleRotate(shift_limit=0.1, 
                                 scale_limit=0.2, 
                                 border_mode=cv2.BORDER_WRAP, 
                                 always_apply=True)(image=img)['image']
        img = A.HorizontalFlip(p=0.5)(image=img)['image']
        
        return img
        
    def merge_images(self, background_img: np.ndarray, foreground_img: np.ndarray) -> np.ndarray:
        mask = self.get_black_mask(foreground_img)
        mask_inv = cv2.bitwise_not(mask)
        
        bg_without_fg = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
        final_img = cv2.add(bg_without_fg, foreground_img)
        
        return final_img

    def get_black_mask(self, img: torch.Tensor, threshold: int = 5) -> torch.Tensor:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        return mask


class SexPositionDataset(TrainDataset):
    """Dataset for 'sex_position' category that inherits ImageDataset
    """    
    
    def get_bg_img(self, idx: int) -> np.ndarray:
        img = self.read_random_bg_img()
        img = self.augment_bg_img(img)
        return img
    
    def get_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        """Create foreground for 'sex position' group. 
        Find all male and female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_name, cur_class = self.get_data_by_idx(idx)
        
        fg_img_paths = glob.glob(os.path.join(self.masks_dir, 'male', f"{img_name}_*"))
        fg_img_paths += glob.glob(os.path.join(self.masks_dir, 'female', f"{img_name}_*"))
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


class TitsSizeDataset(TrainDataset):
    """Dataset for 'tits_size' category that inherits ImageDataset
    """    

    def get_bg_img(self, idx: int) -> np.ndarray:
        
        img = self.read_random_bg_img()
        img = self.augment_bg_img(img)
        
        return img
    
    def get_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        """Create foreground for 'tits_size' group. 
        Find only female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_name, cur_class = self.get_data_by_idx(idx)
        width, height = bg_size
        res_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if cur_class == -1:
            genders = [None]
            if len(glob.glob(os.path.join(self.masks_dir, 'female', f"{img_name}_*"))) > 0:
                genders.append('female')
            if len(glob.glob(os.path.join(self.masks_dir, 'male', f"{img_name}_*"))) > 0:
                genders.append('male')
                
            gender = random.choice(genders)
            if gender is None:
                return res_img
            
            fg_img_paths = glob.glob(os.path.join(self.masks_dir, gender, f"{img_name}_*"))
            fg_imgs = [cv2.imread(path) for path in fg_img_paths]
            for fg_img in fg_imgs:
                fg_img = self.resize_with_pad(fg_img, (width, height))
                res_img += fg_img    
            res_img = self.warp_img(res_img)
            return res_img
            
        fg_img_paths = glob.glob(os.path.join(self.masks_dir, 'female', f"{img_name}_*"))
        fg_imgs = [cv2.imread(path) for path in fg_img_paths]
        for fg_img in fg_imgs:
            fg_img = self.resize_with_pad(fg_img, (width, height))
            res_img += fg_img
        
        res_img = self.warp_img(res_img)
        return res_img


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


def image_to_tensor_train(path):
    tensor = Image.open(path)
    if tensor.size[1] > tensor.size[0] and torch.rand(1) > 0.5:
        tensor = tensor.rotate(90, expand=True)
    return T.ToTensor()(tensor.convert("RGB")).float()  

def image_to_tensor(path):
    tensor = Image.open(path)
    return T.ToTensor()(tensor.convert("RGB")).float()


def get_dataset_by_group(
    group: str,
    foreground_data: pd.DataFrame,
    background_data: pd.DataFrame,
    masks_dir: str,
    pictures_dir: str,
    transforms: nn.Sequential = None,
    train: bool = True,
    badlist_path: str = None,) -> TrainDataset:
    
    if group == 'tits_size':
        return TitsSizeDataset(
            foreground_data,
            background_data,
            masks_dir,
            pictures_dir,
            transforms,
            train,
            badlist_path
        )
        
    if group == 'sex_position':
        return SexPositionDataset(
            foreground_data,
            background_data,
            masks_dir,
            pictures_dir,
            transforms,
            train,
            badlist_path
        )
    
    # default - TitsSizeDataset
    return TitsSizeDataset(
            foreground_data,
            background_data,
            masks_dir,
            pictures_dir,
            transforms,
            train,
            badlist_path
        )

