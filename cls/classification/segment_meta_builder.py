import sys
import os
import json
import torch
import logging
from torch.utils.data import DataLoader
import glob
from pathlib import Path
import math

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.utils.general import save_meta, get_boxes_meta_db, get_meta_id
from cls.classification.engine.datasets import InferenceDBDataset
from cls.classification.engine.augmentation import PreProcess, DataAugmentation
from cls.classification.engine.options import OptionParser
from cls.classification.utils.postgres_db import PostgreSQLHandler

LOGGER = logging.getLogger(__name__)
GENDERS = {'male': 'man', 'female': 'girl'}


def main():
    args = parse_args()
    augmentation = DataAugmentation().eval()
    preprocessing = PreProcess(gray=False, vflip=False, arch="eff")
    
    models_dir = args.models_dir
    datasets_dir = args.datasets_dir
    meta_dir = args.meta_dir
    pictures_dir = args.pictures_dir
    #segments_dir = args.segments_dir
    
    image_groups = args.groups #["sasha test"]
    model_paths = args['MODELS']
    metas = {}
    
    db_handler = PostgreSQLHandler()
    
    for model_group in model_paths:
        for image_group in image_groups:
            LOGGER.info(f'model_group = {model_group}, image_group = {image_group}')
            
            model_path = model_paths[model_group]
            metas = parse_meta_v3(
                model_group, 
                os.path.join(models_dir, model_path), 
                augmentation, 
                preprocessing, 
                datasets_dir, 
                image_group, 
                meta_dir, 
                pictures_dir, 
                db_handler,
                metas,
            )
    
    metas = sort_meta(metas)
    LOGGER.info('Saving meta')
    save_meta(metas, model_paths)


def parse_args():
    parser = OptionParser()
    parser.add_argument('--groups', type=str, nargs='*', default=['group'])
    # parser.add_argument('--host', type=str, default='localhost')
    # parser.add_argument('--database', type=str, default='localhost')
    # parser.add_argument('--user', type=str, default='localhost')
    # parser.add_argument('--password', type=str, default='localhost')
    # parser.add_argument('--port', type=str, default='localhost')
    
    args = parser.parse_args()
    return args 


def parse_meta_v3(
    model_cat: str,
    model_path: str,
    augmentation: DataAugmentation,
    preprocessing: PreProcess,
    datasets_dir: str,
    category: str,
    meta_dir: str,
    pictures_dir: str,
    db_handler: PostgreSQLHandler,
    # segments_dir: str,
    metas: dict,
):

    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(model_path, "cuda", _extra_files=extra_files)
    model = model.eval()
    model.to(torch.float16)
    num2label = json.loads(extra_files['num2label.txt'])

    picset_list = glob.glob(os.path.join(meta_dir, category, '*'))
    
    for picset in picset_list:
        LOGGER.info(f"Picset: {picset}")
        try:
            if picset not in metas.keys():
                metas[picset] = get_meta_id(picset, "meta.json")
            id_meta = metas[picset]
        except:
            LOGGER.info(f"Error with picset: {picset}")
            continue

        list_jpeg = list(id_meta.keys())
        list_jpeg = [os.path.join(pictures_dir, pic) for pic in list_jpeg]
        list_keys = {key.split(".")[0]: key for key in id_meta.keys()}
        dict_boxes = get_boxes_meta_db(list_keys, db_handler)
        
        # list_boxes_jpeg = []
        
        # for pic in dict_boxes:
        #     if os.path.exists(os.path.join(masks_dir, 'female', pic + ".jpeg")):
        #         list_boxes_jpeg.append(os.path.join(masks_dir, 'female', pic + ".jpeg"))
                
        #     if os.path.exists(os.path.join(masks_dir, 'male', pic + ".jpeg")):
        #         list_boxes_jpeg.append(os.path.join(masks_dir, 'male', pic + ".jpeg"))

        # dataset = InferDataset(sorted(list_boxes_jpeg), preprocessing)
        
        image_filenames = []
        
        image_name2fn = {os.path.splitext(fn)[0]: fn for fn in os.listdir(pictures_dir)}
        for pic in list_keys:
            if pic in image_name2fn:
                image_filenames.append(image_name2fn[pic])
            # if os.path.exists(os.path.join(pictures_dir, pic + ".jpeg")): # TODO: not always jpeg
            #     image_filenames.append(pic + ".jpeg")
        
        dataset = InferenceDBDataset(pictures_dir, db_handler, image_filenames, preprocessing)
        loader = DataLoader(dataset, 4, num_workers=32, pin_memory=True, shuffle=False)

        for batch in loader:
            imgs, segments_reprs, img_paths, mask_fns = batch
            
            with torch.no_grad():
                input_tensor = augmentation(imgs.to("cuda"))
                ret_ = model(input_tensor)
                ret_ = torch.sigmoid(ret_)
                ret_ = torch.round(ret_, decimals=2)
                ret_ = ret_.to("cpu")
                
            val, idx = ret_.max(axis=1)
            
            mask_names = tuple(map(lambda x: os.path.splitext(x)[0], mask_fns))
            for num, id_ in enumerate(mask_names):
                if val[num] > 0.5:
                    tag = num2label[str(int(idx[num]))]
                else:
                    tag = model_cat + " trash"
                
                bbox = dict_boxes[id_]["bbox"] #
                cls = GENDERS[dict_boxes[id_]["cls"]]
                origin_name = dict_boxes[id_]["origin"]
                
                image_fn = list_keys[origin_name]
                image_meta = id_meta[image_fn]
                
                id_meta[image_fn] = fill_image_meta(image_meta, tag, bbox, cls, model_cat)
        
        metas[picset] = id_meta

    return metas


def fill_image_meta(image_meta, tag, bbox, cls, model_cat):
    
    # Server receive relative coordinates, so we need to normalize input bbox
    width = float(image_meta['origin']['width'])
    height = float(image_meta['origin']['height'])
    bbox = [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height,
    ]
    
    if "trained" not in image_meta:
        image_meta["trained"] = []

    if len(image_meta["trained"]) > 0 and "bbox" in image_meta["trained"][0]:
        find = False
        for values in image_meta["trained"]:
            if values["bbox"] == bbox:
                values["groups"].append(
                    {
                        "group": model_cat,
                        "category": [" ".join(tag.split("-"))],
                    }
                )
                find = True
                break
        if not find:
            image_meta["trained"].append(
                {
                    "gender": cls,
                    "groups": [
                        {
                            "group": model_cat,
                            "category": [" ".join(tag.split("-"))],
                        }
                    ],
                    "bbox": bbox,
                }
            )
    else:
        image_meta["trained"] = [
            {
                "gender": cls,
                "groups": [
                    {
                        "group": model_cat,
                        "category": [" ".join(tag.split("-"))],
                    }
                ],
                "bbox": bbox,
            }
        ]
        
    return image_meta

def sort_meta(d):
    for key in d:
        for inner_key in d[key]:
            if 'trained' in d[key][inner_key]:
                d[key][inner_key]['trained'].sort(key=lambda x: math.sqrt(sum([i**2 for i in x['bbox'][:2]])))
                for i, item in enumerate(d[key][inner_key]['trained'], start=1):
                    item['idx'] = i
    return d

    
if __name__ == '__main__':
    main()    
