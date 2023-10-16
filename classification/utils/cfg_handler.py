import os
import yaml
from pathlib import Path
from easydict import EasyDict

PROJECT_DIR = str(Path(__file__).parent.parent.parent)
DEFAULT_CFG_PATH = str(Path(PROJECT_DIR) / 'classification' / 'cfg' / 'default.yaml')

def get_cfg(cfg_path: str = DEFAULT_CFG_PATH) -> EasyDict:
    with open(cfg_path) as f:
        cfg = yaml.load(f, yaml.Loader)
        
    cfg['data_path'] = os.path.join(PROJECT_DIR, cfg['data_path'])
    cfg['test_path'] = os.path.join(PROJECT_DIR, cfg['test_path'])
    
    return EasyDict(cfg)
