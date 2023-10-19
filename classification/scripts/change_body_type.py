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
    group = 'body_type'
    dataset_path = os.path.join(cfg.data_path, cfg.datasets_dir, group + '.json')
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    df = pd.read_json(StringIO(data['data']))
    athletic_col = df['athletic']
    bodybuilder_col = df['bodybuilder']
    bodybuilder_col = bodybuilder_col | athletic_col
    df.drop('athletic', axis=1, inplace=True)
    data['data'] = df.to_json()
    
    new_dataset_path = os.path.join(cfg.data_path, cfg.datasets_dir, group + '2.json')
    with open(new_dataset_path, 'w') as f:
        json.dump(data, f)
    

if __name__ == '__main__':
    main()
    
    