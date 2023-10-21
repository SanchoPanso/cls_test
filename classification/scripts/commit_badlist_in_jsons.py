import os
import sys
import json
import pandas as pd
from io import StringIO
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from train.service import TrainWrapper
from utils.cfg_handler import get_cfg


def main():
    cfg = get_cfg()
    segments_dir = os.path.join(cfg.data_path, cfg.segments_dir)
    badlist_path = 'classification/data/badlist_bodytype.txt'
    
    with open(badlist_path) as f:
        bad_fns = f.read().strip().split('\n')
    bad_names = set(map(lambda x: os.path.splitext(x)[0], bad_fns))
    
    for json_fn in os.listdir(segments_dir):
        name = os.path.splitext(json_fn)[0]
        if name not in bad_names:
            continue
        
        print(name)
        json_path = os.path.join(segments_dir, json_fn)
        with open(json_path) as f:
            json_data = json.load(f)
        
        for i in json_data:
            json_data[i]['status'] = 'rejected'
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f)


if __name__ == "__main__":
    main()