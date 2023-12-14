import json
import os
from io import StringIO
import pandas as pd
from os.path import join
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ultralytics import YOLO
import os
import logging
from tqdm import tqdm
import cv2
import numpy as np

from glob import glob

LOGGER = logging.getLogger(__name__)


group_dict = {
    "body_type": [
        "normal body",
        "fat",
        "curvy",
        "skinny",
        "bodybuilder",
        "athletic",
    ],
    "hair_color": [
        "brown hair",
        "blue hair",
        "red hair",
        "violet hair",
        "green hair",
        "blonde hair",
        "black hair",
        "pink hair",
    ],
    "hair_type": ["wavy hair", "curly haired", "straight hair"],
    "sex_positions": [
        "missionary",
        "cowgirl",
        "spooning",
        "doggy style",
        "reverse cowgirl",
        "69",
    ],
    "tits_size": ["huge tits", "small tits", "flat chested", "big tits"],
}


# list af all categories meta.json from my data
def get_all_values():
    all_values = list(chain(*group_dict.values()))
    return all_values


def get_meta(picset, mode="meta.json"):
    if "meta.json" not in picset:
        picset = join(picset, mode)
    with open(picset, "r", encoding="utf-8") as js_f:
        my_js = json.load(js_f)
    return my_js


def get_meta_id(picset, mode=""):
    my_js = get_meta(picset, mode)
    return {
        item["origin"]["filename"]: item
        for item in my_js["items"]
        if "filename" in item["origin"] and len(item["origin"]["filename"]) > 5
    }


# function that calculate accuracy by categorys between two metas json by all categories in group_dict
def calc_acc_by_category(meta1, meta2, all_values):
    acc = defaultdict(lambda: [0, 0])
    for item in meta1:
        cat1 = [category["category"][0] for category in meta1[item]["trained"]]
        cat2 = [category["category"][0] for category in meta2[item]["trained"]]
        for category in all_values:
            if category in cat1 and category in cat2:
                acc[category][0] += 1
            if category not in cat1 and category not in cat2:
                acc[category][0] += 1
            acc[category][1] += 1
    return acc


def parse_meta(
    path_models,
    InferDataset,
    Aug,
    Pre,
    DATASETS,
    CATEGORYS,
    META,
    PICTURE,
    MODE="ret_meta.json",
):
    import torch
    from torch.utils.data import DataLoader
    from glob import glob

    if META is None:
        META = "/home/timssh/ML/TAGGING/DATA/meta"

    metas = {}
    for model_cat, model_path in path_models.items():
        with open(f"{DATASETS}/{model_cat}.json") as json_file:
            json_ = json.load(json_file)
            num2label = json_["num2label"]

        model = torch.jit.load(model_path, "cuda")
        model.eval()
        model.to(torch.float16)

        for cat in CATEGORYS:
            print(cat)
            PICSET_LIST = glob(META + f"/{cat}/*")
            for picset in PICSET_LIST:
                print(picset)
                try:
                    if picset not in metas.keys():
                        metas[picset] = get_meta_id(picset, MODE)
                    id_meta = metas[picset]
                except:
                    continue

                list_jpeg = list(id_meta.keys())
                list_jpeg = [join(PICTURE, pic) for pic in list_jpeg]
                list_keys = {key.split(".")[0]: key for key in id_meta.keys()}

                infer = InferDataset(list_jpeg, Pre)
                loader = DataLoader(infer, 320, num_workers=32, pin_memory=True)

                for batch in loader:
                    with torch.no_grad():
                        ret_ = torch.round(
                            torch.sigmoid(model(Aug(batch[0].to("cuda")))), decimals=2
                        )
                        ret_ = ret_.to("cpu")
                        val, idx = ret_.max(axis=1)
                        for num, id_ in enumerate(batch[1]):
                            if id_ in list_keys:
                                if val[num] > 0.7:
                                    tag = [num2label[str(int(idx[num]))]]
                                else:
                                    tag = []
                                if "trained" in id_meta[list_keys[id_]].keys():
                                    gender_flag = False
                                    for values in id_meta[list_keys[id_]]["trained"]:
                                        if (
                                            "gender" in values.keys()
                                            and values["gender"] == "other"
                                        ):
                                            item = {
                                                "group": model_cat,
                                                "category": [
                                                    " ".join(tg.split("-"))
                                                    for tg in tag
                                                ],
                                            }
                                            values["groups"].append(item)
                                            gender_flag = True
                                            break
                                    if not gender_flag:
                                        id_meta[list_keys[id_]]["trained"].append(
                                            {
                                                "gender": "other",
                                                "group": model_cat,
                                                "category": [
                                                    " ".join(tg.split("-"))
                                                    for tg in tag
                                                ],
                                            }
                                        )

                                else:
                                    id_meta[list_keys[id_]]["trained"] = [
                                        {
                                            "gender": "other",
                                            "group": model_cat,
                                            "category": [
                                                " ".join(tg.split("-")) for tg in tag
                                            ],
                                        }
                                    ]
    return metas


def yolo_proc(yolo_model_path, InferDataset, Pre):
    from ultralytics import YOLO
    import os
    from tqdm import tqdm

    yolo_model = YOLO(yolo_model_path)
    os.makedirs("/home/timssh/ML/TAGGING/DATA/segmentation/picture", exist_ok=True)
    os.makedirs("/home/timssh/ML/TAGGING/DATA/segmentation/boxes", exist_ok=True)

    list_of_paths = glob("/home/timssh/ML/TAGGING/DATA/picture/*")
    list_2_intersect = glob("/home/timssh/ML/TAGGING/DATA/segmentation/boxes/*")
    list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
    list_of_paths = [
        item
        for item in list_of_paths
        if item.split("/")[-1].split(".")[0] not in list_2_intersect
    ]
    infer = InferDataset(list_of_paths, Pre)
    loader = DataLoader(infer, 40, num_workers=32, pin_memory=True, shuffle=False)
    gender_dict = {"girl": "female", "man": "male"}
    for batch in tqdm(loader):
        yolo_results = yolo_model(batch[0])
        for image_id, result in enumerate(yolo_results):
            image_tensor = batch[0][image_id]
            yolo_meta_dict = {}
            for j in range(len(result.boxes.xyxy)):
                if result.boxes.conf[j] >= 0.75:
                    bbox_tensor = result.boxes.xyxy[j].to(torch.float16)
                    bbox_cls = result.boxes.cls[j].to(int)
                    bbox_cls = result.names[int(bbox_cls)]
                    bbox_tensor[0], bbox_tensor[2] = (
                        bbox_tensor[0] / 640,
                        bbox_tensor[2] / 640,
                    )
                    bbox_tensor[1], bbox_tensor[3] = (
                        bbox_tensor[1] / 480,
                        bbox_tensor[3] / 480,
                    )
                    tensor_image = T.ToPILImage()(
                        image_tensor * result.masks.data[j].to("cpu").unsqueeze(0)
                    )
                    tensor_image.save(
                        f"/home/timssh/ML/TAGGING/DATA/segmentation/picture/{batch[1][image_id]}_{gender_dict[bbox_cls]}_{j}.jpeg"
                    )
                    yolo_meta_dict[j] = {
                        "conf": float(result.boxes.conf[j]),
                        "bbox": bbox_tensor.tolist(),
                        "cls": gender_dict[bbox_cls]
                        if bbox_cls in gender_dict.keys()
                        else bbox_cls,
                    }
            if len(yolo_meta_dict) > 0:
                with open(
                    f"/home/timssh/ML/TAGGING/DATA/segmentation/boxes/{batch[1][image_id]}.json",
                    "w",
                ) as f:
                    json.dump(
                        yolo_meta_dict,
                        f,
                    )
    print("done")
    
    
def apply_bilinear_interp(crop: np.ndarray):

    h, w = crop.shape[0], crop.shape[1]
    if w <= 2 or h <= 2:
        return crop

    tmp_img = np.zeros((2, 2, 3), dtype=crop.dtype)
    
    for i in range(2):
        for j in range(2):
            tmp_img[i][j] = crop[i * (h - 1)][j * (w - 1)]
    
    interp_crop = cv2.resize(tmp_img, (w, h), interpolation=cv2.INTER_LINEAR)
    return interp_crop


def yolo_proc_for_img_gen(yolo_model_path, InferDataset, Pre, src_path, dst_path):
    """This is version of 'yolo_proc' for creating dataset for image generation

    :param yolo_model_path: path to yolo model
    :param InferDataset: 
    :param Pre: 
    :param src_path: directory with source images
    :param dst_path: directory in wcich subdirs 'background', 'female', 'male' will be created
    """

    yolo_model = YOLO(yolo_model_path)
    os.makedirs(src_path, exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'female'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'male'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'background'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'boxes'), exist_ok=True)

    list_of_paths = glob(os.path.join(src_path, '*'))
    list_2_intersect = glob(os.path.join(dst_path, 'boxes', '*'))
    list_2_intersect = [item.split("/")[-1].split(".")[0] for item in list_2_intersect]
    
    # Exclude already processed images
    list_of_paths = [
        item
        for item in list_of_paths
        if item.split("/")[-1].split(".")[0] not in list_2_intersect
    ]
        
    # infer = InferDataset(list_of_paths, Pre)
    # loader = DataLoader(infer, 40, num_workers=32, pin_memory=True, shuffle=False)
    gender_dict = {"girl": "female", "man": "male"}
    
    for img_path in tqdm(list_of_paths):
        name, ext = os.path.splitext(os.path.split(img_path)[1])
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        yolo_results = yolo_model(img)
    
        for image_id, result in enumerate(yolo_results):
            yolo_meta_dict = {}
            bg_img = img.copy()
    
            for j in range(len(result.boxes.xyxy)):
                if result.boxes.conf[j] >= 0.75:
                    bbox_tensor = result.boxes.xyxy[j]#.to(torch.float16)
                    bbox_cls = result.boxes.cls[j].to(int)
                    bbox_cls = gender_dict[result.names[int(bbox_cls)]]
                    
                    x1, y1, x2, y2 = map(lambda x: x.int().item(), bbox_tensor.cpu())
                    bg_img[y1: y2, x1: x2] = apply_bilinear_interp(bg_img[y1: y2, x1: x2])
                    
                    mask = result.masks.data[j].cpu().unsqueeze(0).numpy()
                    mask = (mask > 0.5).astype('uint8')[0]
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask = mask[..., np.newaxis]
                    
                    masked_img = img * mask
                    cv2.imwrite(os.path.join(dst_path, bbox_cls, f'{name}_{j}{ext}'), masked_img)
                    
                    yolo_meta_dict[j] = {
                        "conf": float(result.boxes.conf[j]),
                        "bbox": bbox_tensor.tolist(),
                        "cls": gender_dict[bbox_cls]
                        if bbox_cls in gender_dict.keys()
                        else bbox_cls,
                    }
            cv2.imwrite(os.path.join(dst_path, 'background', f'{name}{ext}'), bg_img)
                    
            if len(yolo_meta_dict) > 0:
                with open(os.path.join(dst_path, 'boxes', f"{name}.json"), "w") as f:
                    json.dump(yolo_meta_dict, f)
                    
    print("done")



def get_box(id_item):
    with open(id_item, "r") as json_file:
        json_ = json.load(json_file)
    for key in list(json_.keys()):
        json_[id_item.split("/")[-1].split(".")[0] + "_" + key] = json_.pop(key)
    return json_


def get_boxes_meta(list_keys, bbox_meta):

    list_seg_json = {key.split(".")[0]: key for key in os.listdir(bbox_meta)}
    proc_dict = list_keys.keys() & list_seg_json.keys()
    proc_dict_meta = {}
    for key in proc_dict:
        proc_box = get_box(os.path.join(bbox_meta, list_seg_json[key]))
        for key_ in proc_box.keys():
            proc_box[key_]["origin"] = key
            proc_dict_meta[key_] = proc_box[key_]
    return proc_dict_meta


def parse_meta_v2(
    path_models,
    InferDataset,
    Aug,
    Pre,
    DATASETS,
    CATEGORYS,
    META,
    PICTURE,
    PICTURE_SEG
):
    metas = {}
    for model_cat, model_path in path_models.items():
        with open(f"{DATASETS}/{model_cat}.json") as json_file:
            json_ = json.load(json_file)
            num2label = json_["num2label"]

        model = torch.jit.load(model_path, "cuda")
        model.eval()
        model.to(torch.float16)

        for cat in CATEGORYS:
            print(cat)
            PICSET_LIST = glob(META + f"/{cat}/*")
            for picset in PICSET_LIST:
                print(picset)
                try:
                    if picset not in metas.keys():
                        metas[picset] = get_meta_id(picset, "meta.json")
                    id_meta = metas[picset]
                except:
                    print(picset, "error")
                    continue

                list_jpeg = list(id_meta.keys())
                list_jpeg = [join(PICTURE, pic) for pic in list_jpeg]
                list_keys = {key.split(".")[0]: key for key in id_meta.keys()}
                dict_boxes = get_boxes_meta(list_keys, os.path.join(os.path.dirname(PICTURE), 'segments')) # edited
                
                list_boxes_jpeg = [
                    join(PICTURE_SEG, pic + ".jpg") for pic in dict_boxes.keys() 
                    if os.path.exists(join(PICTURE_SEG, pic + ".jpg"))
                ]
                # list_boxes_jpeg += [
                #     join(PICTURE_SEG, 'male', pic + ".jpeg") for pic in dict_boxes.keys() 
                #     if os.path.exists(join(PICTURE_SEG, 'male', pic + ".jpeg"))
                # ]

                infer = InferDataset(sorted(list_boxes_jpeg), Pre)

                try:
                    samplt = infer[0]
                except Exception as e:
                    print(e)

                loader = DataLoader(
                    infer, 4, num_workers=32, pin_memory=True, shuffle=False
                )

                for batch in loader:
                    with torch.no_grad():
                        ret_ = torch.round(
                            torch.sigmoid(model(Aug(batch[0].to("cuda")))), decimals=2
                        )
                        ret_ = ret_.to("cpu")
                        val, idx = ret_.max(axis=1)
                        # print(val, idx)
                        for num, id_ in enumerate(batch[1]):
                            if val[num] > 0.8:
                                tag = [num2label[str(int(idx[num]))]]
                            else:
                                tag = [model_cat + " trash"]
                            # ids = int(id_.split('_')[-1].split('.')[0])
                            if (
                                len(
                                    id_meta[list_keys[dict_boxes[id_]["origin"]]][
                                        "trained"
                                    ]
                                )
                                > 0
                                and "bbox"
                                in id_meta[list_keys[dict_boxes[id_]["origin"]]][
                                    "trained"
                                ][0].keys()
                            ):
                                find = False
                                for values in id_meta[
                                    list_keys[dict_boxes[id_]["origin"]]
                                ]["trained"]:
                                    if values["bbox"] == dict_boxes[id_]["bbox"]:
                                        values["groups"].append(
                                            {
                                                "group": model_cat,
                                                "category": [
                                                    " ".join(tg.split("-"))
                                                    for tg in tag
                                                ],
                                            }
                                        )
                                        find = True
                                        break
                                if not find:
                                    id_meta[list_keys[dict_boxes[id_]["origin"]]][
                                        "trained"
                                    ].append(
                                        {
                                            "gender": dict_boxes[id_]["cls"],
                                            "groups": [
                                                {
                                                    "group": model_cat,
                                                    "category": [
                                                        " ".join(tg.split("-"))
                                                        for tg in tag
                                                    ],
                                                }
                                            ],
                                            "bbox": dict_boxes[id_]["bbox"],
                                        }
                                    )
                            else:
                                id_meta[list_keys[dict_boxes[id_]["origin"]]][
                                    "trained"
                                ] = [
                                    {
                                        "gender": dict_boxes[id_]["cls"],
                                        "groups": [
                                            {
                                                "group": model_cat,
                                                "category": [
                                                    " ".join(tg.split("-"))
                                                    for tg in tag
                                                ],
                                            }
                                        ],
                                        "bbox": dict_boxes[id_]["bbox"],
                                    }
                                ]

                                """
                                print(ids, id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]), 
                                print(id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"][int(ids)])
                                id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"][int(ids)]["groups"].append(
                                    {   
                                        "group": model_cat,
                                        "category": [
                                            " ".join(tg.split("-")) for tg in tag
                                        ],
                                    }
                                )
                            elif ids > len(id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]):
                                id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"].append(
                                    {   
                                        "gender" : dict_boxes[id_]['cls'],
                                        "groups" : [{"group": model_cat,
                                        "category": [
                                            " ".join(tg.split("-")) for tg in tag
                                        ]}],
                                        'bbox': dict_boxes[id_]['bbox'],
                                    }
                                )
                            else:
                                id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"] = [
                                    {   
                                        "gender" : dict_boxes[id_]['cls'],
                                        "groups" : [{"group": model_cat,
                                        "category": [
                                            " ".join(tg.split("-")) for tg in tag
                                        ]}],
                                        'bbox': dict_boxes[id_]['bbox'],
                                    }
                                ]"""
    return metas


# def save_meta(metas, path_models, mode="meta.json"):
#     for key, value in metas.items():
#         meta_js = get_meta(key, mode)
#         meta_js["items"] = list(value.values())
#         new_neural_versions = {
#             key: value.split("/")[-3] for key, value in path_models.items()
#         }
#         if "neural_version" not in meta_js["picset"].keys():
#             meta_js["picset"]["neural_version"] = new_neural_versions
#         else:
#             for key_ in new_neural_versions.keys():
#                 meta_js["picset"]["neural_version"][key_] = new_neural_versions[key_]
#         print(key)
#         with open(join(key, "ret_meta.json"), "w", encoding="utf-8") as js_f:
#             json.dump(meta_js, js_f, indent=4, ensure_ascii=False)

def save_meta(metas, path_models, mode="meta.json"):
    for key, value in metas.items():
        meta_js = get_meta(key, mode)
        meta_js["items"] = list(value.values())

        for item in meta_js["items"]:
            item['guid'] = item['origin']['filepath'].split('/')[3]
            for i in ['id', 'title', 'text', 'origin', 'thumb', 'tags']:
                if i in item:
                    item.pop(i)
            
            # if 'trained' not in item:
            #     item['trained'] = []

        for pic_key in list(meta_js["picset"].keys()):
            if pic_key in ['guid', 'neural_version']:
                continue
            meta_js["picset"].pop(pic_key)

        new_neural_versions = {
            vkey: 1.0 for vkey, value in path_models.items() # TODO: version
        }
        if "neural_version" not in meta_js["picset"].keys():
            meta_js["picset"]["neural_version"] = new_neural_versions
        else:
            for key_ in new_neural_versions.keys():
                meta_js["picset"]["neural_version"][key_] = new_neural_versions[key_]
        print(key)
        with open(join(key, "ret_meta.json"), "w", encoding="utf-8") as js_f:
            json.dump(meta_js, js_f, indent=4, ensure_ascii=False)


def get_train_noise(PATH, CATEGORY):
    from glob import glob

    TRAIN_LIST = glob(PATH + f"/{CATEGORY}/*/*/picture/*.jpeg")
    jpeg = glob(
        PATH + f"/{CATEGORY}/*/picture/*.jpeg",
    )
    jpg = glob(PATH + f"/{CATEGORY}/*/*/picture/*.jpg")
    jpg1 = glob(PATH + f"/{CATEGORY}/*/picture/*.jpg")

    TRAIN_LIST.extend(jpg)
    TRAIN_LIST.extend(jpg1)
    TRAIN_LIST.extend(jpeg)
    NOISE_LIST = glob(PATH + "/trash/*/picture/*.jpeg")
    n1 = glob(PATH + "/trash/*/*/picture/*.jpeg")
    NOISE_LIST.extend(n1)
    return TRAIN_LIST, NOISE_LIST


def parse_data(list_path, prefix=""):
    index = 4
    parsed = [
        [
            join(prefix, "/".join(item.split("/")[-index:-2])),
            "/".join(item.split("/")[-2:-1]),
            "/".join(item.split("/")[-1:]),
        ]
        for item in list_path
    ]
    return parsed


def class2label(item):
    if "trash" in item:
        return "trash"
    if len(set(item.split("/"))) == len(item.split("/")):
        ret = item.split("/")[1] if len(item.split("/")) > 1 else item
    else:
        ret = item.split("/")[-1]
    return ret


def build_DataFrame(data, PATH):
    import pandas as pd

    df = pd.DataFrame(data)
    df = df.rename({0: "class", 1: "path", 2: "name"}, axis=1)
    df.drop_duplicates(subset=["name"], inplace=True)
    df["label"] = df["class"].apply(class2label)
    df["path"] = df.apply(
        lambda x: join(PATH, x["class"], x["path"], x["name"]), axis=1
    )
    df = df[["path", "label"]]
    df.reset_index(drop=True, inplace=True)
    return df


def build_train_noise(PATH, CATEGORY):
    train, noise = get_train_noise(PATH, CATEGORY)
    train = parse_data(train, prefix=CATEGORY)
    # noise = parse_data(noise, prefix="trash")
    noise = parse_data(noise, prefix="")
    train = build_DataFrame(train, PATH)
    noise = build_DataFrame(noise, PATH)
    return train, noise


def train_val_by_label(df, TEST_SIZE=0.1):
    from sklearn.model_selection import train_test_split

    train, val = [], []
    for uniq_cls in df["label"].unique():
        tr, vl = train_test_split(
            df[df["label"] == uniq_cls], test_size=TEST_SIZE, random_state=234
        )
        train.append(tr)
        val.append(vl)
    return train, val


def get_train_val(df, df_noise, CATEGORY):
    import pandas as pd

    train, val = train_val_by_label(df)
    train_noise, val_noise = train_val_by_label(
        df_noise,
    )

    df_train, df_val = pd.concat(train), pd.concat(val)
    concat_train_list = [df_train, train_noise[0]]
    concat_val_list = [df_val, val_noise[0]]

    train_df, val_df = pd.concat(concat_train_list), pd.concat(concat_val_list)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    train_df.rename({"label": f"{CATEGORY}"}, inplace=True, axis=1)
    val_df.rename({"label": f"{CATEGORY}"}, inplace=True, axis=1)

    val_df, train_df = pd.get_dummies(val_df, columns=[f"{CATEGORY}"]), pd.get_dummies(
        train_df, columns=[f"{CATEGORY}"]
    )
    val_df, train_df = val_df.drop(f"{CATEGORY}_trash", axis=1), train_df.drop(
        f"{CATEGORY}_trash", axis=1
    )
    train_df.columns = [item.replace(f"{CATEGORY}_", "") for item in train_df.columns]
    val_df.columns = [item.replace(f"{CATEGORY}_", "") for item in val_df.columns]
    return train_df, val_df


def build_label(data):
    uniq = data.iloc[:, 1:].columns
    len_uniq = len(uniq)
    range_uniq = range(len_uniq)
    d_ = dict(zip(range_uniq, uniq))
    weights = data.iloc[:, 1:].values.sum() / data.iloc[:, 1:].sum()
    return d_, list(weights)


def save_label(data, num2label, weights, CATEGORY, SOURCE):
    os.makedirs(SOURCE, exist_ok=True)
    with open(SOURCE + f"/{CATEGORY}.json", "w") as f:
        json.dump(
            {
                "cat": [CATEGORY],
                "num2label": num2label,
                f"weights": weights,
                "data": data,
            },
            f,
        )
    LOGGER.info(f"Category `{CATEGORY}` is saved")


def json_builder(train, noise, CATEGORY, SOURCE):
    train_df, val_df = get_train_val(train, noise, CATEGORY)
    num2label, weights = build_label(train_df)
    save_label(
        train_df.to_json(), val_df.to_json(), num2label, weights, CATEGORY, SOURCE
    )
    

def read_dataset_data(dataset_path: str) -> pd.DataFrame:
    with open(dataset_path) as f:
        json_data = json.load(f)
    
    data = pd.read_json(StringIO(json_data['data']))
    return data





