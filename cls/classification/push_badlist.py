import sys
import os
from pathlib import Path
from glob import glob
import cv2
from tqdm import tqdm
import logging
import numpy as np
import json
from ultralytics import YOLO
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.engine.database import PostgreSQLHandler, Picture
from cls.classification.engine.options import OptionParser

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    
    if not os.path.exists(args.badlist_path):
        LOGGER.info(f"badlist_path: {args.badlist_path} does not exist")
        return
    
    LOGGER.info(f"badlist_path: {args.badlist_path}")
    with open(args.badlist_path) as f:
        badlist = f.read().strip().split('\n')
    
    db_handler = PostgreSQLHandler()
    bad_pictures = db_handler.select_picture_by_paths(badlist)
    
    # Update database
    for bad_picture in bad_pictures:
        LOGGER.info(f"Image: {bad_picture.path}")
        bad_picture.status = 'rejected'
        db_handler.update_picture_by_path(bad_picture)
    

def parse_args():
    parser = OptionParser()
    
    parser.add_argument('--badlist_path', type=str, default='cls/classification/data/badlist_common.txt')#default=None)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
