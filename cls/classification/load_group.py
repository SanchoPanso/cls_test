import requests
import json
import pandas as pd
import sys
import os
import logging
import argparse
import asyncio
from pathlib import Path
from easydict import EasyDict

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.engine.options import OptionParser 
from cls.classification.loaders.async_loader import download_images
from cls.classification.loaders.meta_async_loader import get_meta
from cls.classification.utils.general import build_label, save_label

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    
    token = get_token(args.stand)
    data = get_data(args.stand, args.group, token)
    dataset, picset_guids = builder(data)
    
    # Save meta
    asyncio.run(get_meta(token, picset_guids, args.group, args.meta_dir))

    # Save dataset
    full_paths = dataset["path"].to_list()
    dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
    num2label, weights = build_label(dataset)
    save_label(dataset.to_json(), num2label, weights, args.group, args.datasets_dir)
    LOGGER.info(f"File {args.group} was created")

    # Save images
    asyncio.run(download_images(full_paths, args.pictures_dir))
    LOGGER.info("Done")
    

def parse_args() -> EasyDict:
    parser = OptionParser()
    parser.add_argument('--group', type=str, default='tits_size')
    parser.add_argument('--stand', type=str, default='')
    args = parser.parse_args()
    return args


def get_token(stand: str):
    headers = {"Content-Type": "application/json"}
    data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

    url = f"https://yapics.{stand}collect.monster/v1/login"

    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]
    return token


def get_data(stand: str, group: str, token: str):
    url = f"https://yapics.{stand}collect.monster/v1/meta/pictures"
    head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

    groups = {"listBy": 0, "categories": [], "groups": [group], "mode" : ["PREPARING","CHECKING","VALIDATED"]}

    r1 = requests.post(url, data=json.dumps(groups), headers=head, timeout=500000)
    data = r1.json()["data"]
    return data


# function that build 1-hot-encodeing pandas dataset from data
def get_meta_id(data, key):
    my_js = data[key]["pictures"]
    return {item["filepath"]: item for item in my_js}, {str(item["picsetGuid"]) for item in my_js}


def builder(data):
    df_list = []
    picset_guids = set()
    for key in data.keys():
        filenames, guids = get_meta_id(data, key)
        picset_guids.update(guids)
        for filename in filenames.keys():
            if len(filename) < 5:
                continue
            if 'trash' in key:
                df_list.append({"path": filename})
            else:
                df_list.append({"path": filename, key: 1})
    return pd.DataFrame(df_list).fillna(0), {"guids" : list(picset_guids)}


if __name__ == "__main__":
    main()
    