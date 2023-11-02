import sys
from os.path import join

sys.path.append("./CLS/")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from classification.train.augmentation import PreProcess, DataAugmentation
from classification.train.wrapper import get_class_decoder
from insightface.app import FaceAnalysis

from torchvision.transforms import transforms as T
import torch
import gradio as gr
import argparse
import numpy as np
import kornia as K


IMAGE_DIR = "/home/timssh/ML/TAGGING/data"
SOURCE = "/home/timssh/ML/TAGGING/source"


app = FaceAnalysis(
    allowed_modules=[
        "detection",
        "genderage",
    ],
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_thresh=0.7, det_size=(640, 640))


def get_faces(image):
    img = np.array(image)
    faces = app.get(img[:, :, ::-1])
    faces = list(filter(lambda x: x["gender"] == 0, faces))
    if len(faces) == 0:
        return None, None

    faces = sorted(faces, key=lambda x: int(x["bbox"][0]))
    treshes = []
    ages = []
    print(faces)
    for index in range(len(faces[:-1])):
        treshes.append(int(faces[index]["bbox"][2] + faces[index + 1]["bbox"][0]) // 2)
        ages.append(faces[index]["age"])
    treshes.append(None)
    ages.append(faces[-1]["age"])
    curr = 0
    list_of_faces = []
    for tresh in treshes:
        to_tt = T.ToTensor()(img[:, curr:tresh])
        if to_tt.shape[1] > to_tt.shape[2]:
            LMS = 480
        else:
            LMS = 640
        curr = tresh
        to_LMS = K.augmentation.LongestMaxSize(LMS)(to_tt)[0]
        to_480_640 = torch.zeros((3, 480, 640))
        to_480_640[:, : to_LMS.shape[1], : to_LMS.shape[2]] += to_LMS
        list_of_faces.append(to_480_640)

    return list_of_faces, ages


@torch.no_grad()
def get_ret(model, tensor):
    ret_ = torch.round(torch.sigmoid(model(tensor.to("cuda"))), decimals=6)
    return ret_.to("cpu")[0]


def decode_ret(decoder_str, ret_, prefix):
    out = {}
    for i in range(len(decoder_str)):
        out[prefix + decoder_str[str(i)]] = (
            float(ret_[i]) if float(ret_[i]) > 0.9 else 0.0
        )
    return out


def wrap(classes_list):
    def predict(inp):
        with torch.no_grad():
            tensor = ToTensor(inp)
            tensor = Pre(tensor.to(torch.float32))
            tensor = Aug(tensor)
            ret_ = get_ret(model_poses, tensor)
            out = decode_ret(decoder_poses, ret_, "")
            age = []
            face_images, ages = get_faces(inp)
            print(ages)
            if not ages:
                print(1)
                for cat in cats:
                    ret_ = get_ret(models[cat], tensor)
                    out.update(decode_ret(classes_list[cat], ret_, ""))
                return out, ""
            else:
                print(2)
                for index, face in enumerate(face_images):
                    tensor = Pre(face.to(torch.float32))
                    tensor = Aug(tensor)
                    for cat in cats:
                        ret_ = get_ret(models[cat], tensor)
                        out.update(
                            decode_ret(classes_list[cat], ret_, str(index) + " ")
                        )
                    age.append(f"{index}_{ages[index]}")
            return out, "   ".join(age)

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Type path to: model, json, data(optional)"
    )
    parser.add_argument(
        "--model", dest="model", type=str, help="Path to model", required=True
    )
    args = parser.parse_args()

    logger = "wandb"

    cat = args.model.replace(join(SOURCE, logger), "").split("/")[1]
    print(cat)

    cats = [
        "body type",
        "tits size",
        "hair color",
        "hair type",
    ]

    model_paths = {
        "body type": join(
            SOURCE,
            "wandb/body type/version_2_train_eff_32_0.01/checkpoints/epoch=54-step=8140.pt",
        ),
        "tits size": join(
            SOURCE,
            "wandb/tits size/version_0_train_eff_32_0.01/checkpoints/epoch=90-step=11193.pt",
        ),
        "hair color": join(
            SOURCE,
            "wandb/hair color/version_5_train_eff_44_0.01/checkpoints/epoch=50-step=4386.pt",
        ),
        "hair type": join(
            SOURCE,
            "wandb/hair type/version_1_val_eff_44_0.01/checkpoints/epoch=76-step=2156.pt",
        ),
    }

    model_poses = torch.jit.load(
        join(
            SOURCE,
            "wandb/sex positions/version_0_train_eff_44_0.01/checkpoints/epoch=47-step=18384.pt",
        )
    )
    model_poses.to("cuda").eval()
    decoder_poses, _, _, _ = get_class_decoder("sex positions")

    num2label = {}
    models = {}
    zeros = torch.zeros((4, 3, 480, 640)).to("cuda")
    for cat in cats:
        num2label[cat], _, _, _ = get_class_decoder(cat)
        models[cat] = torch.jit.load(model_paths[cat])
        models[cat].to("cuda")
        models[cat].eval()
        models[cat](zeros)

    ###############################
    cat_ex = "hair color"
    _, _, data, _ = get_class_decoder(cat_ex)
    ###############################

    # data = data.sample(100)
    # exemples = []
    # for i in range(len(data)):
    #     ser = data.iloc[i]
    #     exemples.append(ser["dir"])

    Pre = PreProcess(keepdim=False, gray=False, vflip=False, arch="eff")
    Aug = DataAugmentation()
    Aug.eval()
    ToTensor = T.ToTensor()

    gr.Interface(
        fn=wrap(num2label),
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=len(num2label) * 10), "text"],
        # examples=exemples,
    ).launch(share=True)
