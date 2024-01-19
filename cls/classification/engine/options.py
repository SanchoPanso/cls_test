from argparse import ArgumentParser
from collections.abc import Sequence
import os
import sys
import logging
from pathlib import Path
import argparse
from typing import Any
from easydict import EasyDict
from cls.classification.utils.cfg import get_cfg, dict2str

LOGGER = logging.getLogger(__name__)


class OptionParser(argparse.ArgumentParser):
    """summary"""
    
    def __init__(self):
        super().__init__()
        self.add_argument('--cfg', type=str, default=None)
    
    def parse_args(self, args: Sequence[str] | None = None) -> EasyDict:
        args = super().parse_args(args)
        cfg_path = args.cfg
        opts = get_cfg() if cfg_path is None else get_cfg(cfg_path)
        opts.update(vars(args))
        
        LOGGER.info(dict2str(opts))
        
        return opts
    

