import torch
import sys
import os
import numpy as np
import json
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as M
from torchvision.transforms import transforms as T
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, BinaryAccuracy

import pytorch_lightning as pl
from PIL import Image
import pandas as pd
from os.path import join
from typing import List, Dict, Tuple, Sequence

import albumentations as A
import cv2
import numpy as np
import glob 
import random
import math
import logging
from pathlib import Path
from cls.classification.engine.database import PostgreSQLHandler

LOGGER = logging.getLogger(__name__)


class InferenceDirDataset(Dataset):
    def __init__(
        self,
        images_dirs: List[str] | str,
        image_filenames: Sequence[str] = None,
        transforms: nn.Sequential = None,):
    
        self.images_dirs = [images_dirs] if type(images_dirs) == str else images_dirs
        self.transforms = transforms
        
        self.image_paths = []
        
        for imd in images_dirs:
            fns = os.listdir(imd)
            for fn in fns:
                path = os.path.join(imd, fn)
                self.image_paths.append(path)

        if image_filenames is not None:
            image_names = set(map(lambda x: os.path.splitext(x)[0], image_filenames))
            new_image_paths = []
            
            for path in self.image_paths:
                name = os.path.splitext(os.path.basename(path))[0]
                if name in image_names:
                    new_image_paths.append(path)

            self.image_paths = new_image_paths
        
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
    

class InferenceContourDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        segments_dir: str,
        image_filenames: Sequence[str],
        transforms: nn.Sequential = None,):
    
        self.images_dir = images_dir
        self.segments_dir = segments_dir
        self.transforms = transforms
        
        self.mask_fns = []
        self.segments = []
        self.segment_paths = []
        
        for fn in image_filenames:
            name, ext = os.path.splitext(fn)
            segments_path = os.path.join(segments_dir, name + '.json')
            if not os.path.exists(segments_path):
                continue
            
            with open(segments_path) as f:
                segments_data = json.load(f)
            
            for i in segments_data:
                self.mask_fns.append(f"{name}_{i}{ext}")
                self.segments.append(segments_data[i]['segments'])
        
    @torch.no_grad()
    def __getitem__(self, idx):
        
        mask_fn = self.mask_fns[idx]
        mask_name, ext = os.path.splitext(mask_fn)
        name = '_'.join(mask_name.split('_')[:-1])
        img_path = os.path.join(self.images_dir, name + ext)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_with_pad(img, (640, 480))
        img, mask = self.get_segmented_img(img, self.segments[idx])
        img_tr = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img_tr = self.transforms(img_tr)
        
        img_tr = img_tr.squeeze(0).to(torch.float16)
        segments_repr = json.dumps({"segments": self.segments[idx]})
        return img_tr, segments_repr, img_path, mask_fn

    def __len__(self):
        return len(self.mask_fns)
    
    def resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Resize img with pad keeping original ratio"""
        w, h = size
        img = A.LongestMaxSize(max_size=min(w, h))(image=img)['image']
        img = A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT)(image=img)['image']
        return img    

    def get_segmented_img(self, img, segments):
        mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for segment in segments:
            segment = np.array(segment)
            segment = segment.reshape(-1, 1, 2)
            segment[..., 0] *= img.shape[1]
            segment[..., 1] *= img.shape[0]
            segment = segment.astype('int32')
            cv2.fillPoly(mask, [segment], 255)
        
        fg_img = cv2.bitwise_and(img, img, mask=mask)
        return fg_img, mask


class InferenceDBDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        db_handler: PostgreSQLHandler,
        image_filenames: Sequence[str],
        transforms: nn.Sequential = None,):
    
        self.images_dir = images_dir
        self.db_handler = db_handler
        self.transforms = transforms
        
        self.mask_fns = []
        self.segments = []
        self.segment_paths = []
        
        for fn in image_filenames:
            name, ext = os.path.splitext(fn)
            db_pic = self.db_handler.select_picture_by_path(fn)
            if db_pic is None:
                continue
            
            segments_data = db_pic.segments
            for i in segments_data:
                self.mask_fns.append(f"{name}_{i}{ext}")
                self.segments.append(segments_data[i]['segments'])
        
    @torch.no_grad()
    def __getitem__(self, idx):
        
        mask_fn = self.mask_fns[idx]
        mask_name, ext = os.path.splitext(mask_fn)
        name = '_'.join(mask_name.split('_')[:-1])
        img_path = os.path.join(self.images_dir, name + ext)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_with_pad(img, (640, 480))
        img, mask = self.get_segmented_img(img, self.segments[idx])
        img_tr = T.ToTensor()(img).float()
        
        if self.transforms is not None:
            img_tr = self.transforms(img_tr)
        
        img_tr = img_tr.squeeze(0).to(torch.float16)
        segments_repr = json.dumps({"segments": self.segments[idx]})
        return img_tr, segments_repr, img_path, mask_fn

    def __len__(self):
        return len(self.mask_fns)
    
    def resize_with_pad(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Resize img with pad keeping original ratio"""
        w, h = size
        img = A.LongestMaxSize(max_size=min(w, h))(image=img)['image']
        img = A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT)(image=img)['image']
        return img    

    def get_segmented_img(self, img, segments):
        mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for segment in segments:
            segment = np.array(segment)
            segment = segment.reshape(-1, 1, 2)
            segment[..., 0] *= img.shape[1]
            segment[..., 1] *= img.shape[0]
            segment = segment.astype('int32')
            cv2.fillPoly(mask, [segment], 255)
        
        fg_img = cv2.bitwise_and(img, img, mask=mask)
        return fg_img, mask


class GenerativeDataset(Dataset):
    """Dataset for image loading and preprocessing with sample generation.
    This class implements default behaviour of image generation, that can be 
    extended by inherited classes for specific categories.
    """
    
    train: bool
    foreground_data: pd.DataFrame
    background_data: pd.DataFrame
    db_handler: PostgreSQLHandler
    pictures_dir: str
    transforms: nn.Sequential
    num_classes: int
    num2label: Dict[int, str]

    def __init__(
        self,
        foreground_data: pd.DataFrame,
        background_data: pd.DataFrame,
        db_handler: PostgreSQLHandler,
        pictures_dir: str,
        transforms: nn.Sequential = None,
        train: bool = True,
        generativity: float = 0.5,
    ):
        """
        :param data: pandas DataFrame
        :param transforms: image transforms that will be invoked in __getitem__
        :param root: path to dir, defaults to ""
        :param train: whether train or not, defaults to True
        :param group: group of categories that affects the way of image generation, defaults to 'sex_position'
        """

        self.db_handler = db_handler
        badlist = self.get_rejected_paths()
        self.foreground_data = self.delete_badlist(foreground_data.copy(), badlist)
        self.background_data = background_data.copy()
        self.pictures_dir = pictures_dir
        self.train = train
        self.transforms = transforms
        self.generativity = generativity
        
        self.num_classes = len(foreground_data.columns) - 1
        self.num2label = {i: col for i, col in enumerate(foreground_data.columns[1:])}
        
        self.check_foreground_data()
        self.check_background_data()
    
    def check_foreground_data(self):
        self.fg_name2path = {}
        paths = self.foreground_data['path']
        self.fg_segments = {}
        unavailable_ids = []
        for i, path in enumerate(paths):
            name = os.path.splitext(path)[0]
            pic = self.db_handler.select_picture_by_path(path)
            image_path = os.path.join(self.pictures_dir, path)
            
            if pic is None or not os.path.exists(image_path):
                unavailable_ids.append(i)
                continue

            segments = pic.segments
            if not self.check_segments_data(segments):
                unavailable_ids.append(i)
                continue
            
            self.fg_segments[path] = segments
            self.fg_name2path[name] = path 
        
        self.foreground_data.drop(unavailable_ids, axis=0, inplace=True)
        self.foreground_data = self.foreground_data.reset_index(drop=True)
        LOGGER.info(f'Foreground data contains {len(paths)} paths. {len(self.fg_name2path)} is available')
    
    
    def check_background_data(self):
        self.bg_name2path = {}
        paths = self.background_data['path']
        
        unavailable_ids = []
        for i, path in enumerate(paths):
            name = os.path.splitext(path)[0]
            image_path = os.path.join(self.pictures_dir, path)
            
            if os.path.exists(image_path):
               self.bg_name2path[name] = path 
            else:
                unavailable_ids.append(i)
        
        self.background_data.drop(unavailable_ids, axis=0, inplace=True)
        self.background_data = self.background_data.reset_index(drop=True)
        LOGGER.info(f'Background data contains {len(paths)} paths. {len(self.bg_name2path)} is available')

    def get_rejected_paths(self):
        pictures = self.db_handler.select_pictures_by_status('rejected')
        paths = [pic.path for pic in pictures]
        return paths
    
    def check_segments_data(self, segments_data) -> bool:
        file_is_corrupted = False
        for j in segments_data:
            if file_is_corrupted:
                break
            
            for segment in segments_data[j]['segments']:
                if len(segment) % 2 != 0:
                    file_is_corrupted = True
                    break
        return not file_is_corrupted

    @torch.no_grad()
    def __getitem__(self, idx):
        
        if random.random() < self.generativity:
            bg_img = self.get_bg_img(idx)
            bg_w, bg_h = bg_img.shape[1], bg_img.shape[0]
            fg_img, mask = self.get_fg_img(idx, (bg_w, bg_h))
            img = self.merge_images_with_mask(bg_img, fg_img, mask)
        else:
            img = self.get_orig_fg_img(idx)
        
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
    
    def get_fg_img(self, idx: int, bg_size: tuple) -> Tuple[np.ndarray]:
        return self.get_gen_fg_img(idx, bg_size)
    
    def get_orig_fg_img(self, idx: int) -> np.ndarray:
        data = self.foreground_data.iloc[idx]
        img_fn = data['path']
        
        # Note: I dont know how to handle .gif files corectly (cv2.imread cant read this), 
        # that's why I just make black image instead 
        if os.path.splitext(img_fn)[1] == '.gif': 
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.imread(os.path.join(self.pictures_dir, img_fn))
        return img
    
    def get_gen_fg_img(self, idx: int, bg_size: tuple) -> Tuple[np.ndarray]:
        raise NotImplementedError
    
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
        return img_fn, cur_class
    
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
                                 border_mode=cv2.BORDER_CONSTANT, 
                                 always_apply=True)(image=img)['image']
        img = A.HorizontalFlip(p=0.5)(image=img)['image']
        
        return img
    
    def merge_images_with_mask(self, 
                               background_img: np.ndarray, 
                               foreground_img: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        
        mask_inv = cv2.bitwise_not(mask)
        bg_without_fg = cv2.bitwise_and(background_img, background_img, mask=mask_inv)
        final_img = cv2.add(bg_without_fg, foreground_img)
        return final_img
        
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
    
    def get_mask_bbox(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        
        common_contour = np.concatenate(contours, axis=0)
        x, y, w, h = cv2.boundingRect(common_contour)
        return x, y, w, h
    
    def get_segmented_img(self, img, segments):
        mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for segment in segments:
            segment = np.array(segment)
            segment = segment.reshape(-1, 1, 2)
            segment[..., 0] *= img.shape[1]
            segment[..., 1] *= img.shape[0]
            segment = segment.astype('int32')
            cv2.fillPoly(mask, [segment], 255)
        
        fg_img = cv2.bitwise_and(img, img, mask=mask)
        return fg_img, mask


class SceneGenerativeDataset(GenerativeDataset):
    
    def get_bg_img(self, idx: int) -> np.ndarray:
        img = self.read_random_bg_img()
        img = self.augment_bg_img(img)
        return img
    
    def get_gen_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        """Create foreground for 'tits_size' group. 
        Find only female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_fn, cur_class = self.get_data_by_idx(idx)        
        # segment_data = self.db_handler.select_picture_by_path(img_fn)
        segment_data = self.fg_segments[img_fn]
        
        img = cv2.imread(os.path.join(self.pictures_dir, img_fn))
        res_img, mask = self.build_fg_img(img, bg_size, segment_data, cur_class)           
        return res_img, mask
    
    def build_fg_img(self, img: np.ndarray, bg_size: tuple, segment_data: dict, cur_class: int):
        height, width = img.shape[:2]
        res_img = np.zeros((height, width, 3), dtype=np.uint8)
        common_mask = np.zeros((height, width), dtype='uint8')
        
        for i in segment_data:
            segments = segment_data[i]['segments']
            fg_img, mask = self.get_segmented_img(img, segments)
            res_img += fg_img
            common_mask |= mask
        
        rotate_limit = 0 if cur_class != -1 and self.num2label[cur_class] == 'spooning' else 45
        sample = A.ShiftScaleRotate(shift_limit=0.0, 
                                    scale_limit=0.0,
                                    rotate_limit=rotate_limit,
                                    border_mode=cv2.BORDER_CONSTANT, 
                                    always_apply=True)(image=res_img, mask=common_mask)
        res_img = sample['image']
        common_mask = sample['mask']
        
        x, y, w, h = self.get_mask_bbox(common_mask)  
                  
        res_img = res_img[y: y + h, x: x + w]
        res_img = self.resize_with_pad(res_img, bg_size)
        
        common_mask = common_mask[y: y + h, x: x + w] 
        common_mask = self.resize_with_pad(common_mask, bg_size)
        
        return res_img, common_mask


class HumanGenerativeDataset(GenerativeDataset):
        

    def get_bg_img(self, idx: int) -> np.ndarray:
        img = self.read_random_bg_img()
        img = self.augment_bg_img(img)
        return img
    
    def get_gen_fg_img(self, idx: int, bg_size: tuple) -> np.ndarray:
        """Create foreground for 'tits_size' group. 
        Find only female masks for specific index and paste into image with distortions

        :param idx: _description_
        :param bg_size: _description_
        :return: _description_
        """
        img_fn, cur_class = self.get_data_by_idx(idx)
        #segment_data = self.db_handler.select_picture_by_path(img_fn).segments
        segment_data = self.fg_segments[img_fn]
        
        width, height = bg_size
        img = cv2.imread(os.path.join(self.pictures_dir, img_fn))
        
        if cur_class == -1 or self.num2label[cur_class] == 'trash':
            genders = [None]
            classes = [segment_data[i]['cls'] for i in segment_data]
            if 'female' in classes:
                genders.append('female')
            if 'male' in classes:
                genders.append('male')
                
            gender = random.choice(genders)
            if gender is None:
                null_img = np.zeros((height, width, 3), dtype=np.uint8)
                mask = np.zeros((height, width), dtype=np.uint8)
                return null_img, mask
            
            res_img, mask = self.build_fg_img(img, bg_size, segment_data, gender)                       
            return res_img, mask
            
        res_img, mask = self.build_fg_img(img, bg_size, segment_data, 'female')           
        return res_img, mask
    
    def build_fg_img(self, img, bg_size, segment_data, gender):
        height, width = img.shape[:2]
        res_img = np.zeros((height, width, 3), dtype=np.uint8)
        common_mask = np.zeros((height, width), dtype='uint8')
        
        for i in segment_data:
            cls = segment_data[i]['cls']
            if cls != gender:
                continue
            segments = segment_data[i]['segments']
            fg_img, mask = self.get_segmented_img(img, segments)
            res_img += fg_img
            common_mask |= mask
        
        sample = A.ShiftScaleRotate(shift_limit=0.0, 
                                    scale_limit=0.0,
                                    border_mode=cv2.BORDER_CONSTANT, 
                                    always_apply=True)(image=res_img, mask=common_mask)
        res_img = sample['image']
        common_mask = sample['mask']
        
        x, y, w, h = self.get_mask_bbox(common_mask)   
                 
        res_img = res_img[y: y + h, x: x + w]
        res_img = self.resize_with_pad(res_img, bg_size)
        
        common_mask = common_mask[y: y + h, x: x + w] 
        common_mask = self.resize_with_pad(common_mask, bg_size)
        return res_img, common_mask


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
    db_handler: PostgreSQLHandler,
    pictures_dir: str,
    transforms: nn.Sequential = None,
    train: bool = True,
    badlist: List[str] = None,) -> GenerativeDataset:
        
    if group == 'sex_positions':
        return SceneGenerativeDataset(
            foreground_data,
            background_data,
            db_handler,
            pictures_dir,
            transforms,
            train,
        )
    
    return HumanGenerativeDataset(
            foreground_data,
            background_data,
            db_handler,
            pictures_dir,
            transforms,
            train,
        )

