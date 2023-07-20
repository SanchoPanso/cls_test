import pandas as pd
import torch
import json
from torch.nn import functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import Compose
# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip
from timm import create_model
from PIL import Image
from torchvision import transforms as T

def image_to_tensor(path):
    tensor = Image.open(path)
    return T.ToTensor()(tensor.convert("RGB")).float()

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image and label from dataframe
        image = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1:]

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image_to_tensor(image))

        return image, torch.tensor(label)
    
class ImageClassificationModel(LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat)
        # print(y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat)
        # print(y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, sync_dist=True)
        # self.log('val_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

train_transforms = T.Resize(size=(224, 224))
val_transforms = T.Resize(size=(224, 224))

with open("/home/timssh/ML/TAGGING/source_valid/tits_size/tits_size.json") as f:
    json_ = json.load(f)
    cats = json_["cat"]
    num2label = json_["num2label"]
    num_classes = len(json_["weights"])
    weights = json_["weights"]
    train_df, val_df = pd.read_json(json_["train"]), pd.read_json(json_["val"])
# # Load data from pandas dataframes
# train_df = pd.read_csv('train.csv')
# val_df = pd.read_csv('val.csv')

# Create datasets
train_dataset = CustomDataset(train_df, transform=train_transforms)
val_dataset = CustomDataset(val_df, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, num_workers=16 , batch_size=200)
val_loader = DataLoader(val_dataset, num_workers=16 , batch_size=200)

# Create model
model = ImageClassificationModel(num_classes=train_df.shape[1] - 1)
# model = DataParallel(model)
# Create logger
# wandb_logger = WandbLogger(project='image-classification')

# Create trainer
trainer = Trainer(
                  devices=2,
                #   logger=wandb_logger,
                  accelerator="gpu",
                  max_epochs=2,
                  strategy= 'ddp' ,
                # replace_sampler_ddp=False,
                # distributed_backend='gloo'

                  )

# Train model
trainer.fit(model, train_loader, val_loader)