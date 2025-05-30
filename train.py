import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from dotenv import load_dotenv
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config
from rich import print
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10

import wandb
from mmpretrain import get_model

load_dotenv()

wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT"), log_model="all")


pl.seed_everything(7)

lr = 0.1
BATCH_SIZE = 128
n_epochs = 120
DATASET_PATH = Path.cwd().joinpath("data")
data_loader_num_workers = 7

# * ================================
# * data
# * ================================


cifar10_normalization = torchvision.transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)


def split_dataset(dataset, val_split=0.2, train=True):
    """Splits the dataset into train and validation set."""
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(
        dataset, splits, generator=torch.Generator().manual_seed(42)
    )

    if train:
        return dataset_train
    return dataset_val


def get_splits(len_dataset, val_split):
    """Computes split lengths for train and validation set."""
    if isinstance(val_split, int):
        train_len = len_dataset - val_split
        splits = [train_len, val_split]
    elif isinstance(val_split, float):
        val_len = int(val_split * len_dataset)
        train_len = len_dataset - val_len
        splits = [train_len, val_len]
    else:
        raise ValueError(f"Unsupported type {type(val_split)}")

    return splits


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization,
    ]
)
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization,
    ]
)

dataset_train = CIFAR10(
    DATASET_PATH, train=True, download=True, transform=train_transforms
)
dataset_val = CIFAR10(
    DATASET_PATH, train=True, download=True, transform=test_transforms
)
dataset_train = split_dataset(dataset_train)
dataset_val = split_dataset(dataset_val, train=False)
dataset_test = CIFAR10(
    DATASET_PATH, train=False, download=True, transform=test_transforms
)

train_dataloader = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=data_loader_num_workers,
)
val_dataloader = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=data_loader_num_workers,
)
test_dataloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=data_loader_num_workers,
)

# * ================================
# * model
# * ================================


def create_resnet18_cifar10_model():
    model = torchvision.models.resnet18(
        weights=None, num_classes=10
    )  # not pretrained by default
    # smaller kernel size 7-> 3, stride 2-> 1, padding 3-> 1
    # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # remove the maxpool layer # type: ignore

    return model


def create_resnet18_cifar10_model_mmpretrain():
    model = get_model("resnet18_8xb16_cifar10", pretrained=False)
    model.train()  # type: ignore
    return model


def create_piresnet18_cifar10_model():
    pinet_config_filepath = Path.cwd().joinpath(
        "mmpretrain/configs/PiNets/pinet_relu_resnet18_cifar10.py"
    )
    pinet_config = Config.fromfile(pinet_config_filepath)
    model = get_model(pinet_config)
    model.train()  # type: ignore
    return model


class PiNetCif10(LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 60, 80, 100], gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitResnet(PiNetCif10):
    def __init__(self):
        # super().__init__(create_resnet18_cifar10_model_mmpretrain())
        super().__init__(create_piresnet18_cifar10_model())


# * ================================
# * train
# * ================================


net = LitResnet()
wandb_logger.watch(net)
print(
    f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
)

net.configure_optimizers()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    save_last=True,
)

trainer = Trainer(
    max_epochs=n_epochs,
    logger=wandb_logger,  # type: ignore
    callbacks=[checkpoint_callback],
)

trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(net, dataloaders=test_dataloader)

net.save_hyperparameters(ignore=["model"])

artifact = wandb.Artifact("model_weights", type="model")
wandb.log_artifact(artifact)
wandb.save("model.h5")
wandb.finish()
