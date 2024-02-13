import os
import datetime
import argparse
from cls.classification.engine.options import OptionParser
from cls.classification.load_guids import load_guids
from cls.classification.segment_builder import create_segments
from cls.classification.segment_meta_builder import segment_meta_builder


def main():
    args = parse_args()
    pipeline(args.guids, args.yolo_model_path, args.stand)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--guids', nargs='*', default=["9878ccd1-9d83-452c-8e74-e90b9007d074"])
    parser.add_argument('--yolo_model_path', type=str, default='/home/achernikov/CLS/people_models/best_people.pt')
    parser.add_argument('--stand', type=str, default='dev.')  
    args = parser.parse_args()
    return args


def pipeline(guids: str, yolo_model_path: str, stand='dev.'):
    args = OptionParser().parse_args()
    group = get_timestamped_group_name()
    load_guids(guids, stand, group, args)
    create_segments(yolo_model_path, args=args)
    segment_meta_builder([group], args)
    # post_ret_meta()


def get_timestamped_group_name() -> str:
    strftime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    group_name = f'group_{strftime}'
    return group_name


if __name__ == '__main__':
    main()
