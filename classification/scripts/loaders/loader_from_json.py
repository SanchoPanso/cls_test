import requests
import json
import pandas as pd
from pathlib import Path
from os.path import join
import sys

sys.path.append("/home/timssh/ML/TAGGING/CLS/classification/scripts")
import asyncio
from async_handlers.async_loader import main
from utils import build_label, save_label

DATASET_PATH = "/home/timssh/ML/TAGGING/DATA/datasets"
META_PATH = "/home/timssh/ML/TAGGING/DATA/meta"

# stand = 'dev.'
stand = ''

def get_token():
    headers = {"Content-Type": "application/json"}
    data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

    url = f"https://yapics.{stand}collect.monster/v1/login"

    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]
    return token


def load_meta(token, picset_ids):
    url = f"https://yapics.{stand}collect.monster/v1/meta/picsets"
    head = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

    guids = {"guids": picset_ids}

    r1 = requests.post(url, data=json.dumps(guids), headers=head, timeout=500000)
    return r1


def check_groups(data):
    for i in range(1, len(data)):
        # print(data[i-1])
        if data[i]["picset"]["group"] != data[i - 1]["picset"]["group"]:
            return False
    for j in range(0, len(data)):
        path2meta = join(
            META_PATH, data[j]["picset"]["group"][0]["group"], data[j]["picset"]["guid"]
        )
        Path(path2meta).mkdir(parents=True, exist_ok=True)
        with open(join(path2meta, "meta.json"), "w") as f:
            json.dump(data[j], f)
    return True


def buid_dataset(data):
    if not check_groups(data):
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
    # Load the json file with the picset ids
    picset_ids = json.load(
        open(
            "/home/timssh/ML/TAGGING/CLS/classification/scripts/loaders/json2load.json",
            "r",
        )
    )
    # Get the token
    token = get_token()
    # Load the meta
    for picset_id in picset_ids["picsets"]:
        r1 = load_meta(token, picset_id)
        # Build the dataset
        dataset, group = buid_dataset(r1.json()["data"])
        # Save the dataset
        asyncio.run(main(dataset["path"].to_list()))
        dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
        num2label, weights = build_label(dataset)
        save_label(dataset.to_json(), num2label, weights, group, DATASET_PATH)
    print("Done")
