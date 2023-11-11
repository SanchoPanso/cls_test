import wandb
import torch
import json
import os
import sys
import logging
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from cls.classification.engine.options import OptionParser
LOGGER = logging.getLogger(__name__)

def main():
    
    args = parse_args()
    
    run = wandb.init()
    artifact = run.use_artifact(args.artifact_name, type='model')
    root = os.path.join(args.production_path, '/'.join(args.artifact_name.split('/')[1:]))
    artifact_dir = artifact.download(root=root)
    LOGGER.info(f"artifact_dir: {artifact_dir}")

    model_path = os.path.join(artifact_dir, 'model.pt')
    extra_files = {"num2label.txt": ""}  # values will be replaced with data
    model = torch.jit.load(model_path, 'cuda', _extra_files=extra_files)
    num2label = json.loads(extra_files['num2label.txt'])

    LOGGER.info('Model is valid')
    LOGGER.info(f'num2label: {num2label}')


def parse_args():
    parser = OptionParser()
    parser.add_argument('--artifact_name', type=str, default='proektbl-1960/hair_type/model-v__21_train_eff_48_0.001:v1')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()