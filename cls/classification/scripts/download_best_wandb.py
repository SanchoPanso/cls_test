import wandb
import torch
import json
import os
import sys
import logging
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from cls.classification.engine.options import OptionParser
LOGGER = logging.getLogger(__name__)

def main():
    
    args = parse_args()
    
    path = f"{args.entity}/{args.project}"
    LOGGER.info(f"Finding best model in {path}")
    
    best_metric, best_run, best_art = find_best_model(path)
    if best_art is None:
        LOGGER.info(f"Can't find any valid model in project: {args.project}")
        return
    
    LOGGER.info(f"Found model name: {best_art.name}")
    LOGGER.info(f"Metric: {best_metric}")
    
    save_path = os.path.join(args.production_path, f"best_{args.project}")
    LOGGER.info(f"Save in {save_path}")
    best_art.download(root=save_path)
    
    model_path = os.path.join(save_path, 'model.pt')
    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(model_path, 'cuda', _extra_files=extra_files)
    num2label = json.loads(extra_files['num2label.txt'])

    LOGGER.info('Model is valid')
    LOGGER.info(f'num2label: {num2label}')


def parse_args():
    parser = OptionParser()
    parser.add_argument('--entity', type=str, default='proektbl-1960')
    parser.add_argument('--project', type=str, default='hair_type')
    
    args = parser.parse_args()
    return args


def get_max_metric(run, metric_name='val_F1_Macro_epoch'):
    metrics = run.history(samples=10_000, keys=[metric_name], x_axis='epoch')
    max_metric = metrics.values.max(initial=0)
    return max_metric


def get_model_artifact(run, filename='model.pt'):
    arts = run.logged_artifacts()
    valid_arts = []

    for art in arts:
        if art.type != 'model':
            continue

        for file in art.files():
            if file.name == filename:
                valid_arts.append(art)
                break
        
    if len(valid_arts) == 0:
        return None

    return valid_arts[0]


def find_best_model(path='proektbl-1960/hair_type'):
    api = wandb.Api()
    runs = api.runs(path=path)

    best_metric = None
    best_run = None
    best_art = None

    for run in runs:
        LOGGER.info(f"Check run: {run.name}")
        if run.name == 'v__21_train_eff_48_0.001':
            return get_max_metric(run), run, get_model_artifact(run)
        
        art = get_model_artifact(run)
        if art is None:
            continue
        
        metric = get_max_metric(run)
        if best_metric is not None and metric < best_metric:
            continue
        
        best_metric = metric
        best_run = run
        best_art = art
    
    return best_metric, best_run, best_art


if __name__ == '__main__':
    main()