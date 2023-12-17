import sys
import os
import json
import logging
import glob
from pathlib import Path

from cls.classification.engine.options import OptionParser
from cls.classification.utils.postgres_db import PostgreSQLHandler

LOGGER = logging.getLogger(__name__)


def main():
    args = parse_args()
    os.makedirs(args.segments_dir, exist_ok=True)
    db_handler = PostgreSQLHandler()
    pics = db_handler.select_all_pictures()
    
    for pic in pics:
        print(pic.path)
        name = os.path.splitext(pic.path)[0]
        with open(os.path.join(args.segments_dir, name + '.json'), 'w') as f:
            json.dump(pic.segments, f)

def parse_args():
    parser = OptionParser()
    args = parser.parse_args()
    return args     

if __name__ == '__main__':
    main()
    