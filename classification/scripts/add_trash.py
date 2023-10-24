import os
import sys
import json
import numpy as np
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
    group = 'tits_size' #'body_type2'
    dataset_path = os.path.join(cfg.data_path, cfg.datasets_dir, group + '.json')
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    df = pd.read_json(StringIO(data['data']))
    mask = np.ones((len(df),), dtype='bool')
    for col in df.columns[1:]:
        mask &= df[col] == 0
    df['trash'] = mask.astype('int32')
    data['data'] = df.to_json()
    
    new_dataset_path = os.path.join(cfg.data_path, cfg.datasets_dir, group + '_with_trash.json')
    with open(new_dataset_path, 'w') as f:
        json.dump(data, f)
    

if __name__ == '__main__':
    main()
    
    