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

        loss = - einsum("bk...,bk...->", mask, log_p)
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


class FocalTverskyLoss:
    """Focal Tversky Loss for imbalanced 2D and 3D medical image segmentation.

    Combines the Tversky Index (balancing false positives vs false negatives via
    alpha and beta) with a focal parameter (gamma) to focus optimization on hard
    slices or regions that are difficult to segment.
    """

    def __init__(self, *, idk, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 4 / 3, smooth: float = 1e-5,
                 per_image: bool = True, **kwargs):
        self.idk = idk
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.per_image = per_image
        print(
            f"Initialized {self.__class__.__name__} with {idk=}, {alpha=}, "
            f"{beta=}, {gamma=}, {smooth=}, {per_image=}"
        )

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        prediction = pred_softmax[:, self.idk].float()
        target = weak_target[:, self.idk].float()

        dims = tuple(range(2, prediction.ndim)) if self.per_image else (0, *range(2, prediction.ndim))

        tp = (prediction * target).sum(dim=dims)
        fp = (prediction * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - prediction) * target).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.clamp(1.0 - tversky, min=1e-8) ** self.gamma

        return focal_tversky.mean()


FocalTversky = FocalTverskyLoss


def make_loss(name, *, idk):
    losses = {
        "cross_entropy": CrossEntropy,
        "dice": SoftDiceLoss,
        "dice_cross_entropy": DiceCrossEntropy,
        "focal_tversky": FocalTverskyLoss,
        "ftl": FocalTverskyLoss,
    }
    try:
        loss_class = losses[name]
    except KeyError as error:
        raise ValueError(f"Unknown loss: {name}") from error
    return loss_class(idk=idk)


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
