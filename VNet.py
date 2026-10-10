#!/usr/bin/env python3

"""2D V-Net architecture adapted from Milletari et al. (3DV 2016)."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def random_weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


class VNetBlock(nn.Module):
    """Convolutional stage block with residual addition and PReLU activations.

    Applies num_convs convolutions with padding preserved, each followed
    by BatchNorm2d and PReLU. Adds the input via an identity or 1x1 projection
    shortcut before the final PReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(
                    c_in,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.PReLU(out_channels))
        self.convs = nn.Sequential(*layers)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.convs(x) + self.shortcut(x))


class DownStage(nn.Module):
    """Downsampling stage using strided convolution followed by a VNetBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.block = VNetBlock(
            out_channels, out_channels, num_convs=num_convs, kernel_size=kernel_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.down(x))


class UpStage(nn.Module):
    """Upsampling stage using transposed convolution, skip concatenation, and VNetBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.block = VNetBlock(
            out_channels * 2,
            out_channels,
            num_convs=num_convs,
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class VNet(nn.Module):
    """2D V-Net segmentation model with residual learning and PReLU activations.

    Architecture (Milletari et al., 3DV 2016 adapted for 2D):
    - Multi-stage encoder-decoder network
    - Residual connections within each stage
    - Downsampling via strided convolutions (kernel=2, stride=2)
    - Upsampling via transposed convolutions (kernel=2, stride=2)
    - Skip connections between corresponding encoder and decoder levels
    - PReLU activations throughout the network
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kwargs = kwargs

        kernels = kwargs.get("kernels", 16)

        # Stage convolution counts match V-Net: 1, 2, 3, 3, 3, 3, 3, 2, 1
        self.in_stage = VNetBlock(in_dim, kernels, num_convs=1)
        self.down1 = DownStage(kernels, kernels * 2, num_convs=2)
        self.down2 = DownStage(kernels * 2, kernels * 4, num_convs=3)
        self.down3 = DownStage(kernels * 4, kernels * 8, num_convs=3)
        self.down4 = DownStage(kernels * 8, kernels * 16, num_convs=3)

        self.up1 = UpStage(kernels * 16, kernels * 8, num_convs=3)
        self.up2 = UpStage(kernels * 8, kernels * 4, num_convs=3)
        self.up3 = UpStage(kernels * 4, kernels * 2, num_convs=2)
        self.up4 = UpStage(kernels * 2, kernels, num_convs=1)
        self.outc = nn.Conv2d(kernels, out_dim, kernel_size=1)

        print(
            f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}"
        )

    def forward(self, x: Tensor) -> Tensor:
        skip1 = self.in_stage(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.down4(skip4)

        d = self.up1(bottleneck, skip4)
        d = self.up2(d, skip3)
        d = self.up3(d, skip2)
        d = self.up4(d, skip1)

        return self.outc(d)

    def init_weights(self, *args: Any, **kwargs: Any) -> None:
        self.apply(random_weights_init)

