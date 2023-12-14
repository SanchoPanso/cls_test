import os
import sys
import requests
import json
import pandas as pd
from pathlib import Path
import logging
import asyncio
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.loaders.async_loader import download_images
from cls.classification.utils.general import build_label, save_label
from cls.classification.loaders.yapics_api import YapicsAPI
from cls.classification.engine.options import OptionParser

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    
    # Load the json file with the picset ids
    with open(args.json_path) as f:
        picset_ids = json.load(f)
    
    # Get the token
    yapics_api = YapicsAPI(args.stand)
    token = yapics_api.get_token()
    
    # Load the meta
    for picset_id in picset_ids["picsets"]:
        r1 = yapics_api.load_meta(token, picset_id)
    
        # Build the dataset
        dataset, group = buid_dataset(r1.json()["data"], args.meta_dir)
    
        # Save the dataset
        full_paths = dataset["path"].to_list()
        dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
        num2label, weights = build_label(dataset)
        save_label(dataset.to_json(), num2label, weights, group, args.datasets_dir)

        full_paths = yapics_api.get_downloading_urls(full_paths, args.pictures_dir)
        asyncio.run(download_images(full_paths, args.pictures_dir))
        
    LOGGER.info("Done")


def parse_args():
    parser = OptionParser()
    parser.add_argument('--json_path', type=str, 
                        default='/home/achernikov/CLS/cls/classification/data/dev2.json')#os.path.join(os.path.dirname(__file__), "data/json2load_background.json"))
    parser.add_argument('--stand', type=str, default='dev.')
    args = parser.parse_args()
    return args


# def get_token(stand):
#     headers = {"Content-Type": "application/json"}
#     data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

#     url = f"https://yapics2.{stand}collect.monster/v1/login"

#     r = requests.post(url, data=json.dumps(data_log), headers=headers)
#     token = eval(r.text)["token"]
#     return token


# def load_meta(token, picset_ids, stand):
#     url = f"https://yapics2.{stand}collect.monster/v1/meta/picsets"
#     head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

#     guids = {"guids": picset_ids}

#     r1 = requests.post(url, data=json.dumps(guids), headers=head, timeout=500000)
#     return r1


def check_groups(data, meta_dir):
    for i in range(1, len(data)):
        # print(data[i-1])
        if data[i]["picset"]["group"] != data[i - 1]["picset"]["group"]:
            return False
        
    for j in range(0, len(data)):
        
        if "group" in data[j]["picset"]:
            group = data[j]["picset"]["group"][0]["group"]
        else:
            group = 'group' # TODO: find out about that problem

        path2meta = os.path.join(meta_dir, group, data[j]["picset"]["guid"])
        Path(path2meta).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path2meta, "meta.json"), "w") as f:
            json.dump(data[j], f)

    return True


def buid_dataset(data, meta_dir):
    if not check_groups(data, meta_dir):
        raise Exception("Groups are not equal")

    dataset = []
    for picset in data:
        picset_category = (
            picset["picset"]["category"][0]
            if "category" in picset["picset"] and len(picset["picset"]["category"]) > 0
            else "trash"
        )
        for item in picset["items"]:
            if 'filepath' not in item["origin"] or len(item["origin"]["filepath"]) < 5:
                continue
            if "trash" in picset_category:
                dataset.append({"path": item["origin"]["filepath"]})
            else:
                dataset.append({"path": item["origin"]["filepath"], picset_category: 1})
        
    if "group" in data[0]["picset"]:
        group = data[0]["picset"]["group"][0]["group"]
    else:
        group = 'group' # TODO: find out about that problem

    return pd.DataFrame(dataset).fillna(0), group


if __name__ == "__main__":
    main()
