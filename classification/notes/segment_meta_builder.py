import sys

sys.path.append("/home/timssh/ML/TAGGING/CLS/classification")

from scripts.utils import parse_meta_v2, save_meta
from train.model import InferDataset
from train.augmentation import PreProcess, DataAugmentation

Aug = DataAugmentation().eval()
Pre = PreProcess(gray=False, vflip=False, arch="eff")

ROOT_DIR = "/home/timssh/ML/TAGGING/DATA"
MODEL_DIR = f"{ROOT_DIR}/models"
DATASETS = f"{ROOT_DIR}/datasets"
META = f"{ROOT_DIR}/meta"
PICTURE = f"{ROOT_DIR}/picture"


CATEGORYS = [
    # "sex_positions",
    # "tits_size",
    # "hair_color",
    # "hair_type",
    # "body_type",
    # 'body_decoration_body_painting',
    # 'body_decoration_piercing',
    # 'body_decoration_tatto',
    "test"
    # "boobs"
]

path_models = {
    # "body_type": f'{MODEL_DIR}/body_type/version_0_all_eff_36_0.001/checkpoints/epoch=58-step=55460.pt',
    # "tits_size": f'{MODEL_DIR}/tits_size/version_1_train_eff_36_0.01/checkpoints/epoch=65-step=71016.pt',
    # "sex_positions": f'{MODEL_DIR}/sex_positions/version_0_train_eff_32_0.001/checkpoints/epoch=69-step=119560.pt',
    # "hair_color": f'{MODEL_DIR}/hair_color/version_0_train_eff_32_0.001/checkpoints/epoch=121-step=170068.pt',
    # "hair_type": f'{MODEL_DIR}/hair_type/version_0_all_eff_36_0.001/checkpoints/epoch=58-step=53454.pt',
    # "tits_size": f'{MODEL_DIR}/tits_size/v__2_all_eff_36_0.001/checkpoints/epoch=59-step=38280.pt',
    # "body_decoration_body_painting": f"{MODEL_DIR}/body_decoration_body_painting/version_0_train_eff_32_0.001/checkpoints/epoch=65-step=7854.pt",
    # "body_decoration_piercing": f"{MODEL_DIR}/body_decoration_piercing/version_0_train_eff_32_0.001/checkpoints/epoch=9-step=1790.pt",
    # "body_decoration_tatto": f"{MODEL_DIR}/body_decoration_tatto/version_0_train_eff_32_0.001/checkpoints/epoch=38-step=5031.pt",
    "sex_positions": f'{MODEL_DIR}/sex_positions/v__0_train_eff_36_0.001/checkpoints/epoch=52-step=32754.pt',
}


metas = parse_meta_v2(
    path_models, InferDataset, Aug, Pre, DATASETS, CATEGORYS, META, PICTURE
)
save_meta(metas, path_models)