import os
import sys
import json
import pandas as pd
from io import StringIO
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from train.augmentation import DataAugmentation
from train.model import EfficientLightning
from classification.train.wrapper import TrainWrapper
from classification.utils.cfg import get_cfg


def main():
    cfg = get_cfg()
    segments_dir = os.path.join(cfg.data_path, cfg.segments_dir)
    pictures_dir = os.path.join(cfg.data_path, cfg.images_dir)
    correctness_path = os.path.join(cfg.data_path, cfg.correctness_path)
    badlist_path = 'classification/data/badlist_bodytype.txt'
    
    with open(badlist_path) as f:
        bad_fns = f.read().strip().split('\n')
    bad_names = set(map(lambda x: os.path.splitext(x)[0], bad_fns))
    
    fns = []
    statuses = []
    
    for fn in os.listdir(pictures_dir):
        name = os.path.splitext(fn)[0]
        print(name)
        
        # json_path = os.path.join(segments_dir, name + '.json')
        # if not os.path.exists(json_path):
        #     continue
        
        fns.append(name)
        if name in bad_names:
            statuses.append('rejected')
        else:
            statuses.append('approved')    
        
        
    df = pd.DataFrame({'name': fns, 'status': statuses})
    df.to_csv(correctness_path, index=False)


def update_correctness(segments_dir: str, correctness_path: str):
    extra_names = []
    extra_statuses = []
    
    df = pd.read_csv(correctness_path)
    known_paths = set(path for path in df['path'])
    
    for json_fn in os.listdir(segments_dir):
        name = os.path.splitext(json_fn)
        if name not in known_paths:
            extra_names.append(name)
            extra_statuses('unchecked')
    
    extra_df = pd.DataFrame({'name': extra_names, 'status': extra_statuses})
    df = pd.concat([df, extra_df], axis=0, ignore_index=True)
    df.to_csv(correctness_path)


if __name__ == "__main__":
    main()