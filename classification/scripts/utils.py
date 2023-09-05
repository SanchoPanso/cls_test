import json
from os.path import join
from collections import defaultdict
from itertools import chain
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from glob import glob

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
        if len(item["origin"]["filename"]) > 5
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


def parse_meta(path_models, InferDataset, Aug, Pre, DATASETS, CATEGORYS, META, PICTURE):
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
                        metas[picset] = get_meta_id(picset, "meta.json")
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
                                    id_meta[list_keys[id_]]["trained"].append(
                                        {
                                            "group": model_cat,
                                            "category": [
                                                " ".join(tg.split("-")) for tg in tag
                                            ],
                                        }
                                    )
                                else:
                                    id_meta[list_keys[id_]]["trained"] = [
                                        {
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
    list_2_intersect = [item.split('/')[-1].split('.')[0] for item in list_2_intersect]
    list_of_paths = [item for item in list_of_paths if item.split('/')[-1].split('.')[0] not in list_2_intersect]
    infer = InferDataset(list_of_paths, Pre)
    loader = DataLoader(infer, 40, num_workers=32, pin_memory=True, shuffle=False)
    gender_dict = {'girl' : 'female', 'man' : 'male'}
    for batch in tqdm(loader):
        yolo_results = yolo_model(batch[0])
        for image_id, result in enumerate(yolo_results):
            image_tensor = batch[0][image_id]
            yolo_meta_dict = {}
            for j in range(len(result.boxes.xyxy)):
                if result.boxes.conf[j] >= 0.8:
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
                        f"/home/timssh/ML/TAGGING/DATA/segmentation/picture/{batch[1][image_id]}_{j}.jpeg"
                    )
                    yolo_meta_dict[j] = {
                        "conf": float(result.boxes.conf[j]),
                        "bbox": bbox_tensor.tolist(),
                        "cls": gender_dict[bbox_cls] if bbox_cls in gender_dict.keys() else bbox_cls,
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


def get_box(id_item):
    with open(id_item, 'r') as json_file:
        json_ = json.load(json_file)
    for key in list(json_.keys()):
        json_[id_item.split('/')[-1].split('.')[0] + '_' + key] = json_.pop(key)
    return json_

def get_boxes_meta(list_keys):
    import os
    bbox_meta = '/home/timssh/ML/TAGGING/DATA/segmentation/boxes/'
    list_seg_json = { key.split(".")[0]: key for key in os.listdir(bbox_meta)}
    proc_dict = list_keys.keys() & list_seg_json.keys()
    proc_dict_meta = {}
    for key in proc_dict:
        proc_box = get_box(bbox_meta + list_seg_json[key])
        for key_ in proc_box.keys():
            proc_box[key_]['origin'] = key
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
):
    if META is None:
        META = "/home/timssh/ML/TAGGING/DATA/meta"
    PICTURE_SEG = "/home/timssh/ML/TAGGING/DATA/segmentation/picture"
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
                    continue

                list_jpeg = list(id_meta.keys())
                list_jpeg = [join(PICTURE, pic) for pic in list_jpeg]
                list_keys = {key.split(".")[0]: key for key in id_meta.keys()}
                dict_boxes = get_boxes_meta(list_keys)
                list_boxes_jpeg = [join(PICTURE_SEG, pic + '.jpeg') for pic in dict_boxes.keys()]



                infer = InferDataset(sorted(list_boxes_jpeg), Pre)

                samplt = infer[0]

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
                            if val[num] > 0.7:
                                tag = [num2label[str(int(idx[num]))]]
                            else:
                                tag = [model_cat + ' trash']
                            # ids = int(id_.split('_')[-1].split('.')[0])
                            if len(id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]) > 0 and 'bbox' in  id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"][0].keys():
                                find = False
                                for values in id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]:
                                    if values["bbox"] == dict_boxes[id_]['bbox']:
                                        values["groups"].append(
                                            {   
                                                "group": model_cat,
                                                "category": [
                                                    " ".join(tg.split("-")) for tg in tag
                                                ],
                                            }
                                        )
                                        find= True
                                        break
                                if not find:
                                    id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"].append({   
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
                                ]


                                # print(ids, id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]), 
                                # print(id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"][int(ids)])
                            #     id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"][int(ids)]["groups"].append(
                            #         {   
                            #             "group": model_cat,
                            #             "category": [
                            #                 " ".join(tg.split("-")) for tg in tag
                            #             ],
                            #         }
                            #     )
                            # elif ids > len(id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"]):
                            #     id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"].append(
                            #         {   
                            #             "gender" : dict_boxes[id_]['cls'],
                            #             "groups" : [{"group": model_cat,
                            #             "category": [
                            #                 " ".join(tg.split("-")) for tg in tag
                            #             ]}],
                            #             'bbox': dict_boxes[id_]['bbox'],
                            #         }
                            #     )
                            # else:
                            #     id_meta[list_keys[dict_boxes[id_]['origin']]]["trained"] = [
                            #         {   
                            #             "gender" : dict_boxes[id_]['cls'],
                            #             "groups" : [{"group": model_cat,
                            #             "category": [
                            #                 " ".join(tg.split("-")) for tg in tag
                            #             ]}],
                            #             'bbox': dict_boxes[id_]['bbox'],
                            #         }
                            #     ]
    return metas


def save_meta(metas):
    for key, value in metas.items():
        meta_js = get_meta(key)
        meta_js["items"] = list(value.values())
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
    print(CATEGORY)


def json_builder(train, noise, CATEGORY, SOURCE):
    train_df, val_df = get_train_val(train, noise, CATEGORY)
    num2label, weights = build_label(train_df)
    save_label(
        train_df.to_json(), val_df.to_json(), num2label, weights, CATEGORY, SOURCE
    )
