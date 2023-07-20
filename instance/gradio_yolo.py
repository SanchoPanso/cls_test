import gradio as gr
# import torch
from PIL import Image

# # Images
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg', 'zidane.jpg')
# torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg', 'bus.jpg')

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # force_reload=True to update
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/timssh/ML/TAGGING/CLS/instance/runs/segment/train4/weights/best.pt")  # load a pretrained model (recommended for training)

def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference
    # results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results[0].plot())


# inputs = 
# outputs = gr.Image(type="pil")

# examples = [['zidane.jpg'], ['bus.jpg']]
gr.Interface(yolo, gr.Image(type='pil'), gr.Image(type='pil'), analytics_enabled=False).launch(
    share=True)