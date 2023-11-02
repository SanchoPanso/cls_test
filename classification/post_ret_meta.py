import json
import os
import sys
import argparse
import glob
import requests
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).parent))
from utils.cfg import get_cfg, get_opts
from utils.logger import get_logger

LOGGER = get_logger(os.path.splitext(os.path.basename(__file__))[0])


def main():
    
    args = parse_args()
    opts = get_opts(args)
    
    stand = args.stand
    groups = args.groups
    
    url = f"https://yapics.{stand}collect.monster/v1/login"
    headers = {"Content-Type": "application/json"}
    data_log = {"login": "admin", "password": "nC82JpRPLx61901c"}
    r = requests.post(url, data=json.dumps(data_log), headers=headers)
    token = eval(r.text)["token"]

    def get_meta(picset):
        with open(picset, "r", encoding="utf-8") as js_f:
            my_js = json.load(js_f)
        return my_js

    for group in groups:
        paths = glob.glob(os.path.join(opts.meta_dir, group, '*', 'ret_meta.json'))
        for js_path in paths:
            LOGGER.info(f"Group: {group}, json: {js_path}")
            data = get_meta(js_path)
            
            url = f"https://yapics.{stand}collect.monster/v1/picset/trained"
            head = {"Authorization": f"token {token}"}
            r1 = requests.post(url, data=json.dumps(data), headers=head, timeout=500000)
            LOGGER.info(f"Response: {r1.status_code} ({r1.reason}), {r1.text}")
    
            r1 = requests.post(
                f"https://yapics.{stand}collect.monster/v1/picset/checking",
                data=json.dumps({"guids": [js_path.split("/")[-2]]}),
                headers=head,
                timeout=500000,
            )


def parse_args(src_args: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stand', type=str, default='', choices=['dev.', ''])
    parser.add_argument('--groups', type=str, nargs='*', default=['sasha test'])
    
    args = parser.parse_args(src_args)
    return args


if __name__ == "__main__":
    main()


