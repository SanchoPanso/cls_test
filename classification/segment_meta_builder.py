import sys
import os
import json
import torch
from torch.utils.data import DataLoader
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.utils import parse_meta_v2, save_meta, get_boxes_meta, get_meta_id
from train.datasets import InferDataset
from train.augmentation import PreProcess, DataAugmentation
from utils.cfg_handler import get_cfg


def main():
    augmentation = DataAugmentation().eval()
    preprocessing = PreProcess(gray=False, vflip=False, arch="eff")
    
    cfg = get_cfg()
    
    root_dir = cfg.data_path
    models_dir = os.path.join(root_dir, cfg.models_dir)
    datasets_dir = os.path.join(root_dir, cfg.datasets_dir)
    meta_dir = os.path.join(root_dir, cfg.meta_dir)
    pictures_dir = os.path.join(root_dir, cfg.images_dir)
    masks_dir = os.path.join(root_dir, cfg.masks_dir)
    
    # То какие категории мы проверяем в МЕТА
    category = "test"
    group = "tits_size"
    path_model = 'tits_size/v__3_train_eff_48_0.2/checkpoints/epoch=67-step=29716.pt'
    
    # То какими сетками проверяем
    
    os.makedirs(masks_dir, exist_ok=True)
    metas = parse_meta_v3(
        group, 
        os.path.join(models_dir, path_model), 
        augmentation, 
        preprocessing, 
        datasets_dir, 
        category, 
        meta_dir, 
        pictures_dir, 
        masks_dir
    )
    save_meta(metas, {group: path_model})


def parse_meta_v3(
    model_cat,
    model_path,
    augmentation,
    preprocessing,
    datasets_dir,
    category,
    meta_dir,
    pictures_dir,
    masks_dir
):
    
    metas = {}
    with open(f"{datasets_dir}/{model_cat}.json") as json_file:
        json_ = json.load(json_file)
        num2label = json_["num2label"]

    model = torch.jit.load(model_path, "cuda")
    model.eval()
    model.to(torch.float16)

    print(category)
    picset_list = glob.glob(os.path.join(meta_dir, category, '*'))
    
    for picset in picset_list:
        print(picset)
        try:
            if picset not in metas.keys():
                metas[picset] = get_meta_id(picset, "meta.json")
            id_meta = metas[picset]
        except:
            print(picset, "error")
            continue

        list_jpeg = list(id_meta.keys())
        list_jpeg = [os.path.join(pictures_dir, pic) for pic in list_jpeg]
        list_keys = {key.split(".")[0]: key for key in id_meta.keys()}
        dict_boxes = get_boxes_meta(list_keys)
        
        list_boxes_jpeg = []
        
        for pic in dict_boxes:
            if os.path.exists(os.path.join(masks_dir, 'female', pic + ".jpeg")):
                list_boxes_jpeg.append(os.path.join(masks_dir, 'female', pic + ".jpeg"))
                
            if os.path.exists(os.path.join(masks_dir, 'male', pic + ".jpeg")):
                list_boxes_jpeg.append(os.path.join(masks_dir, 'male', pic + ".jpeg"))

        infer = InferDataset(sorted(list_boxes_jpeg), preprocessing)
        loader = DataLoader(infer, 4, num_workers=32, pin_memory=True, shuffle=False)

        for batch in loader:
            with torch.no_grad():
                input_tensor = augmentation(batch[0].to("cuda"))
                ret_ = model(input_tensor)
                ret_ = torch.sigmoid(ret_)
                ret_ = torch.round(ret_, decimals=2)
                ret_ = ret_.to("cpu")
                
            val, idx = ret_.max(axis=1)
            # print(val, idx)
            
            for num, id_ in enumerate(batch[1]):
                if val[num] > 0.8:
                    tag = num2label[str(int(idx[num]))]
                else:
                    tag = model_cat + " trash"
                
                bbox = dict_boxes[id_]["bbox"]
                cls = dict_boxes[id_]["cls"]
                origin_name = dict_boxes[id_]["origin"]
                
                image_fn = list_keys[origin_name]
                image_meta = id_meta[image_fn]
                
                fill_image_meta(image_meta, tag, bbox, cls, model_cat)

    return metas


def fill_image_meta(image_meta, tag, bbox, cls, model_cat):
    
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


    
if __name__ == '__main__':
    main()    
