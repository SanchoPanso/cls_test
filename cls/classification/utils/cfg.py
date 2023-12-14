import os
import yaml
import logging
import argparse
from pathlib import Path
from easydict import EasyDict

LOGGER = logging.getLogger(__name__)

PROJECT_DIR = str(Path(__file__).parent.parent.parent.parent)
DEFAULT_CFG_PATH = str(Path(PROJECT_DIR) / 'cls' / 'classification' / 'cfg' / 'default.yaml')


def get_cfg(cfg_path: str = DEFAULT_CFG_PATH) -> EasyDict:
    with open(cfg_path) as f:
        raw_cfg = yaml.load(f, yaml.Loader)
    
    cfg = {}
    
    for cfg_category in raw_cfg:
        
        if cfg_category == 'MODELS':
            cfg.update({'MODELS': raw_cfg['MODELS']})
            continue    
        
        cfg_category_path = PROJECT_DIR
        if 'PATH' in raw_cfg[cfg_category]:
            cfg_category_path = os.path.join(PROJECT_DIR, raw_cfg[cfg_category]['PATH'])
        
        for key in raw_cfg[cfg_category]:
            if key == 'PATH':
                cfg[cfg_category.lower() + "_path"] = cfg_category_path
                continue
            
            cfg[key] = os.path.join(cfg_category_path, raw_cfg[cfg_category][key])
    
    return EasyDict(cfg)


def get_opts(args: argparse.Namespace, cfg_path: str = None) -> EasyDict:
    cfg_path = DEFAULT_CFG_PATH if cfg_path is None else cfg_path
    cfg = get_cfg(cfg_path)
    opts = cfg
    opts.update(vars(args))
    
    LOGGER.info(f'Options: {dict2str(opts)}')
    return opts 


def dict2str(d: dict) -> str:
    value_reprs = []
    first_half_min_len = 25
    for key in d:
        value_repr = f"{' ' * max(0, first_half_min_len - len(key))}{key} : {d[key]}"
        value_reprs.append(value_repr)
    
    res = '\n'.join(value_reprs)
    return res

def dict2str(d: dict) -> str:
    value_reprs = []
    for key in d:
        value_repr = f"{key}={d[key]}"
        value_reprs.append(value_repr)
    
    res = ', '.join(value_reprs)
    return res

