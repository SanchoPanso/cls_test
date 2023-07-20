import torch
import torch.nn as nn
import kornia as K


class PreProcess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    def __init__(self, keepdim=True, gray=False, vflip=True, arch="eff") -> None:
        super().__init__()
        self.gray = gray
        self.vflip = vflip
        self.arch = arch

        if self.arch == "vit":
            self.preproc = K.augmentation.PadTo((256, 256), keepdim=keepdim)
        elif self.arch == "vit_l_16":
            self.preproc = K.augmentation.PadTo((512, 512), keepdim=keepdim)
        else:
            self.preproc = K.augmentation.Resize((480, 640), keepdim=keepdim)

        self.resize1 = K.augmentation.LongestMaxSize(256)
        self.resize2 = K.augmentation.LongestMaxSize(512)
        self.to_gray = K.augmentation.RandomGrayscale(p=0.3, keepdim=keepdim)
        self.to_vflip = K.augmentation.RandomVerticalFlip(
            p=0.5,
            keepdim=True,
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "vit":
            x = self.resize1(x)[0]

        if self.arch == "vit_l":
            x = self.resize2(x)[0]

        x_out = self.preproc(x)
        if self.gray:
            x_out = self.to_gray(x_out)
        if self.vflip:
            x_out = self.to_vflip(x_out)

        return x_out


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms_train = nn.Sequential(
            K.augmentation.RandomErasing(
                scale=(0.02, 0.03),
                ratio=(0.1, 0.15),
                p=0.4,
            ),
            K.augmentation.RandomHorizontalFlip(
                p=0.5,
            ),
            K.augmentation.RandomAffine(
                degrees=40,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                p=0.8,
            ),
            K.augmentation.RandomGaussianBlur(
                (3, 3),
                (0.1, 2.0),
                p=0.4,
            ),
            K.augmentation.RandomPerspective(
                0.4,
                p=0.3,
            ),
            K.augmentation.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )
        self.transforms_val = nn.Sequential(
            K.augmentation.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.transforms_train(x)  # BxCxHxW
        else:
            x = self.transforms_val(x)
        return x
