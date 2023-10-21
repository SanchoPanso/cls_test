from PIL import Image
from ultralytics import YOLO
import os
from torchvision.transforms import transforms as T
import kornia as K
from tqdm import tqdm

PATH = "/home/timssh/ML/TAGGING/CLS/instance/runs/segment/train9/weights/best.pt"
model = YOLO(PATH)

masks = '/home/timssh/ML/TAGGING/DATA/masks'
image_path = '/home/timssh/ML/TAGGING/DATA/picture'
pic_list = os.listdir(image_path)

# TODO: Переписать на более быстрый варианm с использованием батчей
for pic in tqdm(pic_list):
    if pic in os.listdir(masks):
        continue
    image = Image.open(os.path.join(image_path, pic)).convert('RGB')
    img = T.ToTensor()(image)
    if img.size()[1] < img.size()[2]:                    
        img = K.augmentation.LongestMaxSize(640)(img)
    else:
        img = K.augmentation.LongestMaxSize(480)(img)
    img = K.augmentation.PadTo((480, 640), keepdim=True)(img.squeeze(0))
    res = model.predict(T.ToPILImage()(img))
    for result in res:
        if len(result.boxes) == 0:
            T.ToPILImage()(img).save(os.path.join(masks, pic))
            continue
        flag = False
        for mask, bbox in zip(result.masks, result.boxes):
            if int(bbox.cls) < 0.5:
                mask_img = K.augmentation.PadTo((480, 640), keepdim=True)(
                    img[:]
                    * mask[:].data.to("cpu")
                )
                T.ToPILImage()(mask_img).save(os.path.join(masks, pic))
                flag = True
                break
        if not flag:
            mask_img = K.augmentation.PadTo((480, 640), keepdim=True)(
                    img[:]
                    * mask[:].data.to("cpu")
                )
            T.ToPILImage()(mask_img).save(os.path.join(masks, pic))