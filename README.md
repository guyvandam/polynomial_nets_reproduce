# Experiment Reproduction of [*Deep Polynomial Neural Networks*](https://ieeexplore.ieee.org/document/9353253)

More specifically, the experiment reproduction of **Table 5 - ResNet18 on CIFAR10 image classification task.** with the reported training setup w/ pytorch lightning and mmpretrain.


> **Accuracy**: *accruacy* refers to *top-1 accuracy*


# ⚠️ Problems
[ ] the reported 94% accuracy is not achieved. Currently at 92% \
[ ] #paramters of the prodpoly is 5.2M instead of the reported 6M - might signal a **bug** in this implementation.

# 1. Implementing the baseline w/ *the reported training setup*
The baseline ResNet18 on CIFAR10 with pytorch lightning in this [link](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html) is reported to get 93-94% accuracy on CIFAR10 with 40-50 epochs with [their learning rate scheduler.](https://arc.net/l/quote/ghcogmmt)

Using the above guide, with the simple lr scheduler and batch size reported in the [paper](https://ieeexplore.ieee.org/document/9353253)
 
 > "Each method is trained for 120 epochs with batch size 128. The SGD optimizer is used with initial learning rate of 0.1. The learning rate is multiplied with a factor of 0.1 in epochs 40; 60; 80; 100."

[p.7 on pdf, just above the tables 4. and 5] achived **88%** accuracy though the paper claims ~94%.

(I assume) The paper uses an mmpretrain implementation of ResNet as described in this [folder](https://github.com/grigorisg9gr/polynomial_nets/tree/master/classification-NO-activation-function) more specifically this [passage](https://arc.net/l/quote/nfjpacqa)

The discrepancy could be explained by the difference in the native pytorch implementation and the mm implementation.







## Difference in native pytorch implementation and mm implementation
That difference is mainly where the relus are located in the backbone and a identity path that appears in the mm implementation but not in the pytorch one. See [diff](https://www.diffchecker.com/PATnnqym/) that was generate with the [mmpretrain_v_lightning.py](mmpretrain_v_lightning.py) script.

> ✅ Using the mmpretrain implementation with `get_model` and the reported training setup in the paper achived **92%** accuracy.

## mmpretrain installation
You want to install mm from source as you'll need to modify the backbone to the PiNet version:
1.install it from source - [link](https://mmpretrain.readthedocs.io/en/stable/get_started.html#install-from-source)

> **Important**: You can usee my [forked version](https://github.com/guyvandam/mmpretrain) of mmpretrain as the backbones are already modified. You can look at the commits. They are simple and few. 

1. install the mmcv with [mim](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-mim-recommended) (2 commands :)


# 2. Implementing ProdPoly
The modification to the mm ResNet backbones are placed in this [folder](https://github.com/grigorisg9gr/polynomial_nets/tree/master/classification-NO-activation-function).

I've used the [pinet_relu.py](https://github.com/grigorisg9gr/polynomial_nets/blob/master/classification-NO-activation-function/backbones/pinet_relu.py) backbone modification, which yeilds (from what I understand) a **2nd order polynomial expansion** you can see the [diff - orig | pinet relu](https://www.diffchecker.com/nZxIzRHs/) between the [baseline mmpretrain backbone](https://github.com/open-mmlab/mmpretrain/blob/master/mmcls/models/backbones/resnet.py) and the modified [pinet backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/classification-NO-activation-function/backbones/pinet_relu.py)

Adds (what I assume is) the instance normalization as norm3 and the Hadamard product in lines 135-136 to yeild a 2nd order polynomial expansion (see the `second` variable)

## Modification to mmpretrain
1. Table 5 in the paper the Prodpoly resnet has [2, 2, 1, 1] residual blocks. Those changes are made in the PiResNet backbone in this [commit](https://github.com/open-mmlab/mmpretrain/commit/2f5ccd8b3736bbc475cce7fb0fd4f83f96136a99) as a custom depth in `arch_settings` and for the config file that is added in the following [commit](https://github.com/open-mmlab/mmpretrain/commit/53de2b2275a5f4d85181afa40737c2733f36e7c0#diff-dcc2b901a5be9c157b4eeae867edc6901e2989b276d5c8124de2debce68b1c95) - notice the `"18pi"` depth.

### Adding a custom backbone
2. Used this [very simple guide](https://mmpretrain.readthedocs.io/en/stable/advanced_guides/modules.html#add-a-new-backbone) to add the custom pinet backbone with the changes described above and seen in the those commits.

> Notice the `resnet18_8xb16_cifar10` config uses a `ResNet_CIFAR` backbone.


You can also see the diff between the baseline and the pinet model both implemented in mmpretrain with script in the comparisons folder. And its [diff](https://www.diffchecker.com/oALIyNbY/)

