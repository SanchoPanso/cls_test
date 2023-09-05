import gradio as gr
import torch
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
from diffusers import StableDiffusionLatentUpscalePipeline


model_2x = "stabilityai/sd-x2-latent-upscaler"
upscaler2x = StableDiffusionLatentUpscalePipeline.from_pretrained(
    model_2x, torch_dtype=torch.float16
)
upscaler2x.to("cuda")


def get_new_image_shape(image):
    width, height = image.size
    new_width = width - (width % 32)
    new_height = height - (height % 32)
    return new_width, new_height


def upscale(raw_img, scale=5, steps=10):
    # raw_img = Image.open(raw_img).convert("RGB")
    old_width, old_height = raw_img.size
    new_width, new_height = get_new_image_shape(raw_img)
    if old_width != new_width or old_height != new_height:
        raw_img = raw_img.resize((new_width, new_height))
    upscaled_image = upscaler2x(
        prompt='',
        negative_prompt='',
        image=raw_img,
        guidance_scale=scale,
        num_inference_steps=steps,
    ).images[0]
    upscaled_image = upscaled_image.resize((old_width, old_height))
    return upscaled_image


# # Images
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg', 'zidane.jpg')
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg', 'bus.jpg')

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # force_reload=True to update
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    "/home/timssh/ML/TAGGING/CLS/instance/runs/segment/train7/weights/best.pt"
)  # load a pretrained model (recommended for training)


def yolo(image):
    # g = (size / max(im.size))  # gain
    # im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize
    
    im = image.copy()

    # if im.size[0] < 380 or im.size[1] < 380:
    #     im = upscale(im)
    # im = im.resize((640,480))
    # image = image.resize((640,480))

    image = T.ToTensor()(image)
    results = model(im)  # inference
    # results.render()  # updates results.imgs with boxes and labels

    results = results[0].plot(conf=0.5)
    return Image.fromarray(results)

    # zeros = torch.zeros_like(image)
    # for result in results:
    #     if result.boxes.conf[0] >= 0.5:
    #         for mask in result.masks:
    #             zeros += mask.data.to('cpu')
    # image = image * zeros
    # return T.ToPILImage()(image)

# inputs =
# outputs = gr.Image(type="pil")

# examples = [['zidane.jpg'], ['bus.jpg']]
gr.Interface(
    yolo, gr.Image(type="pil"), gr.Image(type="pil"), analytics_enabled=False
).launch(share=True)
