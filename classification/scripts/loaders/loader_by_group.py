import requests
import json
import pandas as pd
import sys

sys.path.append("/home/timssh/ML/TAGGING/CLS/classification/scripts")
import asyncio
from async_handlers.async_loader import main
from async_handlers.meta_async_loader import get_meta
from utils import build_label, save_label


DATASET_PATH = "/home/timssh/ML/TAGGING/DATA/datasets"

# Choose group
group = "sex_positions"
# Choose stand
stand = ""


def get_token():
    headers = {"Content-Type": "application/json"}
    data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}

    url = f"https://yapics.{stand}collect.monster/v1/login"

    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]
    return token


def get_data():
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
    token = get_token()
    data = get_data()
    dataset, picset_guids = builder(data)
    asyncio.run(get_meta(token, picset_guids, group))
    asyncio.run(main(dataset["path"].to_list()))
    dataset["path"] = dataset["path"].apply(lambda x: x.split("/")[-1])
    num2label, weights = build_label(dataset)
    save_label(dataset.to_json(), num2label, weights, group, DATASET_PATH)
    print(f"File {group} was created")
