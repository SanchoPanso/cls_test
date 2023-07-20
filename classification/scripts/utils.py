import json
from os.path import join
from collections import defaultdict
from itertools import chain

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
        item["origin"]["filename"] : item
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
                # list_jpeg = glob(picset + "/picture/*.jpeg")
                # list_jpeg.extend(glob(picset + "/picture/*.jpg"))
                list_jpeg = list(id_meta.keys())
                list_jpeg = [join(PICTURE, pic) for pic in list_jpeg]
                list_keys = {key.split('.')[0] : key for key in id_meta.keys()}
                infer = InferDataset(list_jpeg, Pre)
                loader = DataLoader(infer, 320, 
                                    num_workers=32, pin_memory=True
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
                            # if id_ in id_meta.keys():
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
