import os
import yaml
from pathlib import Path
from easydict import EasyDict

DEFAULT_CFG_PATH = str(Path(__file__).parent.parent / 'cfg' / 'default.yaml')

def get_cfg(cfg_path: str = DEFAULT_CFG_PATH) -> EasyDict:
    with open(cfg_path) as f:
        cfg = yaml.load(f, yaml.Loader)
    return cfg
