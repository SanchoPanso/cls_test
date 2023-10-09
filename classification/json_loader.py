import requests
import json
import pandas as pd
from pathlib import Path
from os.path import join
import os
import sys
import asyncio
import argparse

sys.path.append(str(Path(__file__).parent))
from loaders.async_loader import download_images
from utils.utils import build_label, save_label
from classification.utils.cfg_handler import get_cfg

def main():
    cfg = get_cfg()
    args = parse_args()
    
    datasets_dir = os.path.join(cfg['data_path'], cfg['datasets_dir'])
    meta_dir = os.path.join(cfg['data_path'], cfg['meta_dir'])
    images_dir = os.path.join(cfg['data_path'], cfg['images_dir'])
    
    # Load the json file with the picset ids
    with open(args.json_path) as f:
        picset_ids = json.load(f)
    
    # Get the token
    token = get_token(args.stand)
    
    # Load the meta
    for picset_id in picset_ids["picsets"]:
        r1 = load_meta(token, picset_id, args.stand)
    
        # Build the dataset
        dataset, group = buid_dataset(r1.json()["data"], meta_dir)
    
        # Save the dataset
        asyncio.run(download_images(dataset["path"].to_list(), images_dir))
        dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
        num2label, weights = build_label(dataset)
        save_label(dataset.to_json(), num2label, weights, group, datasets_dir)
    
    print("Done")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default="./classification/json_data/json2load.json")
    parser.add_argument('--stand', type=str, default='')
    args = parser.parse_args()
    return args

def get_token(stand):
    headers = {"Content-Type": "application/json"}
    data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

    url = f"https://yapics.{stand}collect.monster/v1/login"

    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]
    return token


def load_meta(token, picset_ids, stand):
    url = f"https://yapics.{stand}collect.monster/v1/meta/picsets"
    head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

    guids = {"guids": picset_ids}

    r1 = requests.post(url, data=json.dumps(guids), headers=head, timeout=500000)
    return r1


def check_groups(data, meta_dir):
    for i in range(1, len(data)):
        # print(data[i-1])
        if data[i]["picset"]["group"] != data[i - 1]["picset"]["group"]:
            return False
    for j in range(0, len(data)):
        path2meta = join(
            meta_dir, data[j]["picset"]["group"][0]["group"], data[j]["picset"]["guid"]
        )
        Path(path2meta).mkdir(parents=True, exist_ok=True)
        with open(join(path2meta, "meta.json"), "w") as f:
            json.dump(data[j], f)
    return True


def buid_dataset(data, meta_dir):
    if not check_groups(data, meta_dir):
        raise Exception("Groups are not equal")

    dataset = []
    for picset in data:
        picset_category = (
            picset["picset"]["category"][0]
            if len(picset["picset"]["category"]) > 0
            else "trash"
        )
        for item in picset["items"]:
            if len(item["origin"]["filepath"]) < 5:
                continue
            if "trash" in picset_category:
                dataset.append({"path": item["origin"]["filepath"]})
            else:
                dataset.append({"path": item["origin"]["filepath"], picset_category: 1})
    return pd.DataFrame(dataset).fillna(0), data[0]["picset"]["group"][0]["group"]


if __name__ == "__main__":
    main()