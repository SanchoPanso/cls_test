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
from typing import List, Dict

import albumentations as A
import cv2
import numpy as np
import glob 
import os
import random
import math


class ImageDataset(Dataset):
    """Dataset for image loading and preprocessing with sample generation.
    This class implements default behaviour of image generation, that can be 
    extended by inherited classes for specific categories.
    """
    root: str
    train: bool
    transforms: nn.Sequential
    num_classes: int
    num2label: Dict[int, str]

    def __init__(
        self,
        data: pd.DataFrame,
        transforms: nn.Sequential = None,
        root: str = "",
        train: bool = True,
        badlist_path: str = None,
    ):
        """
        :param data: pandas DataFrame
        :param transforms: image transforms that will be invoked in __getitem__
        :param root: path to dir that must contain subdirs 'background', 'male', 'female', defaults to ""
        :param train: whether train or not, defaults to True
        :param group: group of categories that affects the way of image generation, defaults to 'sex_position'
        """
        
        if badlist_path is not None:
            with open(badlist_path) as f:
                text = f.read().strip()
                badlist = [] if text == '' else text.split('\n')
            self.data = self.delete_badlist(data, badlist)
        else:
            self.data = data
            
        self.root = root
        self.train = train
        self.transforms = transforms
        
        self.num_classes = len(data.columns) - 1
        self.num2label = {i: col for i, col in enumerate(data.columns[1:])}

    @torch.no_grad()
    def __getitem__(self, idx):
        
        # Get random background img from "{root}/background/"
        bg_img = self.get_bg_img(idx)
        
        # Get foreground img by index using dirs "{root}/male/" and "{root}/female/"
        fg_img = self.get_fg_img(idx, (bg_img.shape[1], bg_img.shape[0]))
        
        img = self.merge_images(bg_img, fg_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        # Random decrease of th resolution for training
        if self.train and torch.rand(1) > 0.7:
            img = self.resize_tensor_with_pad(img, (320, 240))
        img = self.resize_tensor_with_pad(img, (640, 480))
        
        img = img.squeeze(0).to(torch.float16)
        label = torch.tensor(self.data.iloc[idx].tolist()[1:], dtype=torch.float16)
        return img, label

    def __len__(self):
        return len(self.data)
    
    @classmethod
    def delete_badlist(cls, df: pd.DataFrame, badlist: List[str]) -> pd.DataFrame:
        
        df.index = df[df.columns[0]]        
        clear_badlist = []
        for b in badlist:
            if b in df.index:
                clear_badlist.append(b)

        df = df.drop(clear_badlist)
        index = range(len(df))
        df.index = index
        return df
        
    
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
        
        fg_img_paths = glob.glob(os.path.join(self.root, 'male', f"{img_name}_*"))
        fg_img_paths += glob.glob(os.path.join(self.root, 'female', f"{img_name}_*"))
        fg_imgs = [cv2.imread(path) for path in fg_img_paths]
        
        width, height = bg_size
        res_img = np.zeros((height, width, 3), dtype='uint8')
        for fg_img in fg_imgs:
            fg_img = self.resize_with_pad(fg_img, (width, height))
            res_img += fg_img
        
        res_img = self.warp_img(res_img)
        return res_img
    
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
    
    def read_random_bg_img(self) -> np.ndarray:
        """Read random background image listed in self.data"""
        bg_paths = glob.glob(os.path.join(self.root, 'background', '*'))
        idx = random.randint(0, len(bg_paths) - 1)
        img_path = bg_paths[idx]
        
        # Note: I dont know how to handle .gif files corectly (cv2.imread cant read this), 
        # that's why I just make black image instead 
        if os.path.splitext(img_path)[1] == '.gif': 
            img = np.zeros((640, 640, 3), dtype='uint8')
        else:
            img = cv2.imread(img_path)
        return img
    
    def augment_bg_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        crop_h, crop_w = max(1, h // 2), max(1, w // 2)
        img = A.RandomCrop(crop_h, crop_w, p=0.5)(image=img)['image']
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


class SexPositionDataset(ImageDataset):
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


class TitsSizeDataset(ImageDataset):
    """Dataset for 'sex_position' category that inherits ImageDataset
    """    

    def __init__(
        self,
        data: pd.DataFrame,
        transforms: nn.Sequential = None,
        root: str = "",
        train: bool = True,
        badlist_path: str = None,
    ):
        super().__init__(data, transforms, root, train, badlist_path)
        self.prepare_data()
    
    def prepare_data(self):
        trash_mask = np.ones((len(self.data),), dtype='bool')
        cols = self.data.columns.tolist()
        
        for i in range(1, len(cols)):
            col = cols[i]
            trash_mask &= (self.data[col] == 0).values

        trash_data = self.data[trash_mask]
        expanded_trash_data = []
        
        new_columns = ['trash_bg', 'trash_male', 'trash_female']
        expanded_cols = trash_data.columns.tolist() + new_columns
        
        for i in range(len(trash_data)):
            path = trash_data['path'].iloc[i]
            name, ext = os.path.splitext(path)
            
            bg_row = {col: 0 for col in expanded_cols}
            bg_row['path'] = path
            bg_row['trash_bg'] = 1
            expanded_trash_data.append(bg_row)
            
            female_paths = glob.glob(os.path.join(self.root, 'female', f'{name}*'))
            if len(female_paths) > 0:
                female_row = {col: 0 for col in expanded_cols}
                female_row['path'] = path
                female_row['trash_male'] = 1
                expanded_trash_data.append(female_row)
            
            male_paths = glob.glob(os.path.join(self.root, 'male', f'{name}*'))
            if len(male_paths) > 0:
                male_row = {col: 0 for col in expanded_cols}
                male_row['path'] = path
                male_row['trash_female'] = 1
                expanded_trash_data.append(male_row)
            
        expanded_trash_data = pd.DataFrame(expanded_trash_data)
        non_trash_data = self.data[~trash_mask]
        
        for col in new_columns:
            non_trash_data[col] = [0] * len(non_trash_data)
        
        self.data = pd.concat([non_trash_data, expanded_trash_data], axis=0, ignore_index=True)
        self.num_classes += 3
        for col in new_columns:
            self.num2label[len(self.num2label)] = col
        
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
        res_img = np.zeros((height, width, 3), dtype='uint8')
        
        if self.num2label[cur_class] == 'trash_bg':
            return res_img
        
        if self.num2label[cur_class] == 'trash_female':
            fg_img_paths = glob.glob(os.path.join(self.root, 'female', f"{img_name}_*"))
            fg_imgs = [cv2.imread(path) for path in fg_img_paths]
            for fg_img in fg_imgs:
                fg_img = self.resize_with_pad(fg_img, (width, height))
                res_img += fg_img    
            res_img = self.warp_img(res_img)
            return res_img
        
        if self.num2label[cur_class] == 'trash_male':
            fg_img_paths = glob.glob(os.path.join(self.root, 'male', f"{img_name}_*"))
            fg_imgs = [cv2.imread(path) for path in fg_img_paths]
            for fg_img in fg_imgs:
                fg_img = self.resize_with_pad(fg_img, (width, height))
                res_img += fg_img 
            res_img = self.warp_img(res_img)
            return res_img
            
        fg_img_paths = glob.glob(os.path.join(self.root, 'female', f"{img_name}_*"))
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
    data: pd.DataFrame,
    transforms: nn.Sequential = None,
    root: str = "",
    train: bool = True,
    badlist_path: str = None) -> ImageDataset:
    
    available_groups = ['sex_position', 'tits_size']
    assert group in available_groups
    
    if group == 'tits_size':
        return TitsSizeDataset(data, transforms, root, train, badlist_path)
    if group == 'sex_position':
        return SexPositionDataset(data, transforms, root, train, badlist_path)
    
    # default - ImageDataset
    return ImageDataset(data, transforms, root, train, badlist_path)



