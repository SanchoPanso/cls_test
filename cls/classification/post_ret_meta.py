import json
import os
import sys
import argparse
import glob
import logging
import requests
from pathlib import Path
from typing import Sequence, List

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.engine.options import OptionParser
from cls.classification.loaders.yapics_api import YapicsAPI

LOGGER = logging.getLogger(__name__)


def main():
    
    args = parse_args()
    
    stand = args.stand
    groups = args.groups
    
    post_ret_meta(stand, groups, args)


def parse_args(src_args: Sequence[str] | None = None):
    parser = OptionParser()
    parser.add_argument('--stand', type=str, default='dev.', choices=['dev.', ''])
    parser.add_argument('--groups', type=str, nargs='*', default=['group'])
    
    args = parser.parse_args(src_args)
    return args


def post_ret_meta(stand: str, groups: List[str], args):
    yapics_api = YapicsAPI(args.stand)
    token = yapics_api.get_token()

    for group in groups:
        paths = glob.glob(os.path.join(args.meta_dir, group, '*', 'ret_meta.json'))
        
        for js_path in paths:
            LOGGER.info(f"Group: {group}, json: {js_path}")
            
            data = get_meta(js_path)
            r1 = yapics_api.post_trained(token, data)
            LOGGER.info(f"Response: {r1.status_code} ({r1.reason}), {r1.text}")

            r1 = yapics_api.set_checking(token, js_path)
            LOGGER.info(r1)


def get_meta(picset):
    with open(picset, "r", encoding="utf-8") as js_f:
        my_js = json.load(js_f)
    return my_js


if __name__ == "__main__":
    main()
