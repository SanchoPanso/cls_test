import sys
from os.path import join

sys.path.append("./CLS/")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from classification.train.augmentation import PreProcess, DataAugmentation
from classification.train.service import get_class_decoder
from ultralytics import YOLO
from torchvision.transforms import transforms as T
import torch
import gradio as gr
import numpy as np
import kornia as K


SOURCE = "/home/timssh/ML/TAGGING/DATA/datasets"
MODELS = "/home/timssh/ML/TAGGING/DATA/models"

yolo_model = YOLO(
    "/home/timssh/ML/TAGGING/CLS/instance/runs/segment/train/weights/best.pt"
)


def get_yolo(image):
    list_of_crops = []
    img = image.resize((640, 480))
    res = yolo_model(img)
    for result in res:
        for mask, bbox in zip(result.masks, result.boxes):
            r = bbox.xyxy[0].to(int)
            if bbox.conf > 0.5:
                list_of_crops.append(
                    [
                        K.augmentation.PadTo((480, 640), keepdim=True)(
                            T.ToTensor()(np.array(img)[r[1] : r[3], r[0] : r[2]])
                            * mask.data[:, r[1] : r[3], r[0] : r[2]].to("cpu")
                        ),
                        result.names[int(bbox.cls[0])],
                    ]
                )

    return list_of_crops


@torch.no_grad()
def get_ret(model, tensor):
    ret_ = torch.round(torch.sigmoid(model(tensor.to("cuda"))), decimals=6)
    return ret_.to("cpu")[0]


def decode_ret(decoder_str, ret_, prefix):
    out = {}
    i = int(ret_.argmax())
    out[prefix + decoder_str[str(i)]] = float(ret_[i]) if float(ret_[i]) > 0.75 else 0.0
    return out


def get_porn_cat(list_of_actors):
    ret = [key.split("_")[1] for key in list_of_actors]
    sret = set(ret)
    if len(ret) == 1:
        return "girl solo   " if list(ret)[0] == "girl" else "man solo   "
    else:
        if len(sret) == 2:
            return "straight    "
        else:
            return "lesb    " if list(ret)[0] == "girl" else "gay   "


def wrap(classes_list):
    def predict(inp):
        with torch.no_grad():
            tensor = ToTensor(inp)
            tensor = Pre(tensor.to(torch.float32))
            tensor = Aug(tensor)
            ret_ = get_ret(model_poses, tensor)
            out = decode_ret(decoder_poses, ret_, "")
            gender = []
            list_of_crops = get_yolo(inp)
            if len(list_of_crops) > 0:
                for index, crop in enumerate(list_of_crops):
                    tensor = Pre(crop[0].to(torch.float32))
                    tensor = Aug(tensor)
                    for cat in cats:
                        ret_ = get_ret(models[cat], tensor)
                        if crop[1] == "girl":
                            out.update(
                                decode_ret(classes_list[cat], ret_, str(index) + " ")
                            )
                    gender.append(f"{index}_{crop[1]}")

            return out, get_porn_cat(gender) + "   ".join(gender)

    return predict


if __name__ == "__main__":
    logger = "wandb"

    cats = [
        "body_type",
        "tits_size",
        "hair_color",
        "hair_type",
        "body_decoration_tatto",
        "body_decoration_piercing",
        "body_decoration_body_painting",
    ]

    model_paths = {
        "body_type": join(
            MODELS,
            "body_type/version_2_train_eff_32_0.01/checkpoints/epoch=54-step=8140.pt",
        ),
        "tits_size": join(
            MODELS,
            "tits_size/version_1_train_eff_36_0.01/checkpoints/epoch=65-step=71016.pt",
        ),
        "hair_color": join(
            MODELS,
            "hair_color/version_0_train_eff_32_0.001/checkpoints/epoch=121-step=170068.pt",
        ),
        "hair_type": join(
            MODELS,
            "hair_type/version_0_train_eff_32_0.001/checkpoints/epoch=138-step=137332.pt",
        ),
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

    poses = "sex_positions"
    model_poses = torch.jit.load(
        join(
            MODELS,
            f"{poses}/version_0_train_eff_32_0.001/checkpoints/epoch=69-step=119560.pt",
        )
    )
    model_poses.to("cuda").eval()
    decoder_poses, _ = get_class_decoder(poses, SOURCE)

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
        outputs=[gr.Label(num_top_classes=len(num2label) * 10), "text"],
    ).launch(share=True)
