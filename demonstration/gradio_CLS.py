import sys
from os.path import join

sys.path.append("./CLS/")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from engine.augmentation import PreProcess, DataAugmentation
from engine.wrapper import get_class_decoder

from torchvision.transforms import transforms as T
import torch
import gradio as gr


# IMAGE_DIR = "/home/timssh/ML/TAGGING/data"
# SOURCE = "/home/timssh/ML/TAGGING/source/source_valid"
# for new version used IMADE_DIR and SOURCE from code below
SOURCE = "/home/timssh/ML/TAGGING/DATA/datasets"
MODELS = "/home/timssh/ML/TAGGING/DATA/models"


def wrap(classes_list):
    def predict(inp):
        out = {}
        tensor = ToTensor(inp)
        tensor = Pre(tensor.to(torch.float32))
        tensor = Aug(tensor)
        with torch.no_grad():
            for cat in cats:
                ret_ = get_ret(models[cat], tensor)
                out.update(decode_ret(classes_list[cat], ret_, ""))
        return out

    return predict


@torch.no_grad()
def get_ret(model, tensor):
    ret_ = torch.round(torch.sigmoid(model(tensor.to("cuda"))), decimals=6)
    return ret_.to("cpu")[0]


def decode_ret(decoder_str, ret_, prefix):
    out = {}
    for i in range(len(decoder_str)):
        out[prefix + decoder_str[str(i)]] = (
            float(ret_[i]) if float(ret_[i]) > 0.75 else 0.0
        )
    return out


if __name__ == "__main__":
    logger = "wandb"

    cats = [
        # "body type",
        # "tits_size",
        # "hair color",
        # "hair type",
        # "body_decorating"
        "body_decoration_tatto",
        "body_decoration_piercing",
        "body_decoration_body_painting",
    ]

    model_paths = {
        # "body type": join(
        #     SOURCE,
        #     "wandb/body type/version_2_train_eff_32_0.01/checkpoints/epoch=54-step=8140.pt",
        # ),
        # "tits size": join(
        #     SOURCE,
        #     "wandb/tits size/version_0_train_eff_32_0.01/checkpoints/epoch=90-step=11193.pt",
        # ),
        # "tits_size": join(
        #     SOURCE,
        #     "wandb/tits_size/version_1_train_eff_36_0.01/checkpoints/epoch=65-step=71016.pt",
        # ),
        # "hair color": join(
        #     SOURCE,
        #     "wandb/hair color/version_5_train_eff_44_0.01/checkpoints/epoch=50-step=4386.pt",
        # ),
        # "hair type": join(
        #     SOURCE,
        #     "wandb/hair type/version_1_val_eff_44_0.01/checkpoints/epoch=76-step=2156.pt",
        # ),
        # "body_decorating" : join(
        #     SOURCE,
        #     "wandb/body_decorating/version_0_train_eff_32_0.001/checkpoints/epoch=21-step=3784.pt",
        # ),
        "body_decoration_body_painting": join(
            MODELS,
            "body_decoration_body_painting/version_0_train_eff_32_0.001/checkpoints/epoch=65-step=7854.pt",
        ),
        "body_decoration_piercing": join(
            MODELS,
            "body_decoration_piercing/version_0_train_eff_32_0.001/checkpoints/epoch=9-step=1790.pt",
        ),
        "body_decoration_tatto": join(
            MODELS,
            "body_decoration_tatto/version_0_train_eff_32_0.001/checkpoints/epoch=38-step=5031.pt",
        ),
    }

    num2label = {}
    models = {}
    zeros = torch.zeros((4, 3, 480, 640)).to("cuda")
    for cat in cats:
        num2label[cat], _ = get_class_decoder(cat, SOURCE)
        models[cat] = torch.jit.load(model_paths[cat])
        models[cat].to("cuda")
        models[cat].eval()
        models[cat](zeros)


    Pre = PreProcess(keepdim=False, gray=False, vflip=False, arch="eff")
    Aug = DataAugmentation()
    Aug.eval()
    ToTensor = T.ToTensor()

    gr.Interface(
        fn=wrap(num2label),
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=len(num2label) * 10)],
    ).launch(share=True)
