import os
import datetime
import argparse
from cls.classification.engine.options import OptionParser
from cls.classification.load_guids import load_guids
from cls.classification.segment_builder import create_segments
from cls.classification.segment_meta_builder import segment_meta_builder
from cls.classification.post_ret_meta import post_ret_meta
from cls.classification.loaders.yapics_api import YapicsAPI


def main():
    args = parse_args()
    pipeline(args.guids, args.yolo_model_path, args.stand)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--guids', nargs='*', default=["4aa26a8b-985e-4e09-b843-29b112d93797"])#["9ea527b3-c5e2-4aa3-9899-094db57f620e"])
    parser.add_argument('--yolo_model_path', type=str, default='/data/achernikov/workspace/model_manager/best.pt')
    parser.add_argument('--stand', type=str, default='dev.')  
    args = parser.parse_args()
    return args


def pipeline(guids: str, yolo_model_path: str, stand='dev.'):
    args = OptionParser().parse_args([])
    group = get_timestamped_group_name()
    set_preparing(guids, stand)
    load_guids(guids, stand, group, args)
    create_segments(yolo_model_path, args=args)
    segment_meta_builder([group], args)
    post_ret_meta(stand, [group], args)


def set_preparing(guids, stand):
    api = YapicsAPI(stand)
    token = api.get_token()
    api.set_preparing(token, guids)


def get_timestamped_group_name() -> str:
    strftime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    group_name = f'group_{strftime}'
    return group_name


if __name__ == '__main__':
    main()
