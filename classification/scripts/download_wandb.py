import wandb
import torch
import json
import os
import sys
from pathlib import Path


artifact_name = 'proektbl-1960/hair_type/model-v__21_train_eff_48_0.001:v1'#'proektbl-1960/hair_type/model-v__3_all_eff_48_0.001:v0'

run = wandb.init()
artifact = run.use_artifact(artifact_name, type='model')
artifact_dir = artifact.download()
print(artifact_dir)

model_path = os.path.join(artifact_dir, 'model.pt')
extra_files = {"num2label.txt": ""}  # values will be replaced with data
model = torch.jit.load(model_path, 'cuda', _extra_files=extra_files)
num2label = json.loads(extra_files['num2label.txt'])

print(model)
print(num2label)

# args = parse_args()
# cfg = get_cfg(args.cfg)
# cfg.update(vars(args))

# WRAPPER = TrainWrapper(
#     cfg=cfg,
#     num_workers=cfg.num_workers,
# )

# model = EfficientLightning(
#     model=WRAPPER.model,
#     num2label=WRAPPER.num2label,
#     batch_size=WRAPPER.batch_size,
#     decay=WRAPPER.decay,
#     augmentation=DataAugmentation(),
#     weights=WRAPPER.weights,
# )

# checkpoint = torch.load('./artifacts/model-v__3_all_eff_48_0.001:v0/model.ckpt')
# state_dict = checkpoint['state_dict']
# model.load_state_dict(state_dict)

# print(model)

