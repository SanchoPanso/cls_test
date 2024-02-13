import os
import sys
import requests
import json
import pandas as pd
from pathlib import Path
from typing import List
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
    load_guids(args.guids, args.stand, args.group, args)
    

def parse_args():
    parser = OptionParser()
    parser.add_argument('--guids', nargs='*', default=[]) 
    parser.add_argument('--stand', type=str, default='dev.')
    parser.add_argument('--group', type=str, default='group')
    args = parser.parse_args()
    return args


def load_guids(guids: List[str], stand: str, group: str, args: argparse.Namespace):
    
    # Get the token
    yapics_api = YapicsAPI(stand)
    token = yapics_api.get_token()
    
    # Load the meta
    for guid in guids:
        r1 = yapics_api.load_meta(token, [guid])
    
        # Build the dataset
        dataset, group = buid_dataset(r1.json()["data"], args.meta_dir, group)
    
        # Save the dataset
        full_paths = dataset["path"].to_list()
        dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
        num2label, weights = build_label(dataset)
        save_label(dataset.to_json(), num2label, weights, group, args.datasets_dir)

        full_paths = yapics_api.get_downloading_urls(full_paths, args.pictures_dir)
        asyncio.run(download_images(full_paths, args.pictures_dir))
        
    LOGGER.info("Done")



def check_groups(data, meta_dir, group):
    # for i in range(1, len(data)):
        # print(data[i-1])
        # if data[i]["picset"]["group"] != data[i - 1]["picset"]["group"]:
        #     return False
        
    for j in range(0, len(data)):
        
        if "group" in data[j]["picset"]:
            group = data[j]["picset"]["group"][0]["group"]
        else:
            group = group # TODO: find out about that problem

        path2meta = os.path.join(meta_dir, group, data[j]["picset"]["guid"])
        Path(path2meta).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path2meta, "meta.json"), "w") as f:
            json.dump(data[j], f)

    return True


def buid_dataset(data, meta_dir, group_name='group'):
    if not check_groups(data, meta_dir, group_name):
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
        group = group_name # TODO: find out about that problem

    return pd.DataFrame(dataset).fillna(0), group


if __name__ == "__main__":
    main()
