#!/usr/bin/env python3

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    # Adapted from source architecture: https://arxiv.org/abs/1505.04597
    # We are using the original U-Net architecture, but we include padding, so
    # the dimension of the output mask is the same as the input image.
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kwargs = kwargs

        kernels = kwargs.get("kernels", 64)

        self.inc = DoubleConv(in_dim, kernels)
        self.down1 = MaxPool(kernels, kernels * 2)
        self.down2 = MaxPool(kernels * 2, kernels * 4)
        self.down3 = MaxPool(kernels * 4, kernels * 8)
        self.down4 = MaxPool(kernels * 8, kernels * 16)

        self.up1 = UpConv(kernels * 16, kernels * 8)
        self.up2 = UpConv(kernels * 8, kernels * 4)
        self.up3 = UpConv(kernels * 4, kernels * 2)
        self.up4 = UpConv(kernels * 2, kernels)
        self.outc = nn.Conv2d(kernels, out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)


def random_weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sequence = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.sequence(x)


class MaxPool(nn.Module):
    def __init__(self, in_chan, out_chan, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_chan, out_chan)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_chan, out_chan, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.up = nn.ConvTranspose2d(in_chan, in_chan // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_chan, out_chan)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)