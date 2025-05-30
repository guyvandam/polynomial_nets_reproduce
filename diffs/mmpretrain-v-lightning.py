# https://www.diffchecker.com/PATnnqym/
import pyperclip
import torch
import torch.nn as nn
import torchvision

from mmpretrain import get_model

lit_model = torchvision.models.resnet18(
    weights=None, num_classes=10
)  # not pretrained by default

# mofity to accomodate the cifar10 dataset
# smaller kernel size 7-> 3, stride 2-> 1, padding 3-> 1
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
lit_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
lit_model.maxpool = nn.Identity()  # remove the maxpool layer # type: ignore

pyperclip.copy(str(lit_model))

mmpretrain_model = get_model("resnet18_8xb16_cifar10", pretrained=False)
# keep the backbone only for easier comparison!
mmpretrain_model = mmpretrain_model.backbone  # type: ignore

pyperclip.copy(str(mmpretrain_model))
