# https://www.diffchecker.com/oALIyNbY/
from pathlib import Path

import pyperclip
from mmengine.config import Config

from mmpretrain import get_model

baseline_model = get_model("resnet18_8xb16_cifar10", pretrained=False)
pyperclip.copy(str(baseline_model))

pinet_config_filepath = Path.cwd().joinpath(
    "mmpretrain/configs/PiNets/pinet_relu_resnet18_cifar10.py"
)
pinet_config = Config.fromfile(pinet_config_filepath)
pinet_model = get_model(pinet_config)

pyperclip.copy(str(pinet_model))
