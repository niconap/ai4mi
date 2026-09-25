#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class SoftDiceLoss:
    """Multiclass soft Dice loss computed from probabilities."""

    def __init__(self, *, idk, smooth=1e-5):
        self.idk = idk
        self.smooth = smooth
        print(f"Initialized {self.__class__.__name__} with {idk=}, {smooth=}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        prediction = pred_softmax[:, self.idk].float()
        target = weak_target[:, self.idk].float()
        intersection = (prediction * target).sum(dim=(0, 2, 3))
        denominator = (prediction + target).sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class DiceCrossEntropy:
    """Weighted combination of the existing cross-entropy and soft Dice losses."""

    def __init__(self, *, idk, dice_weight=0.5):
        self.cross_entropy = CrossEntropy(idk=idk)
        self.dice = SoftDiceLoss(idk=idk)
        self.dice_weight = dice_weight
        print(f"Initialized {self.__class__.__name__} with {dice_weight=}")

    def __call__(self, pred_softmax, weak_target):
        dice_weight = self.dice_weight
        return ((1 - dice_weight) * self.cross_entropy(pred_softmax, weak_target)
                + dice_weight * self.dice(pred_softmax, weak_target))


def make_loss(name, *, idk):
    losses = {
        "cross_entropy": CrossEntropy,
        "dice": SoftDiceLoss,
        "dice_cross_entropy": DiceCrossEntropy,
    }
    try:
        loss_class = losses[name]
    except KeyError as error:
        raise ValueError(f"Unknown loss: {name}") from error
    return loss_class(idk=idk)


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
