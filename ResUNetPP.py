#!/usr/bin/env python3

"""A 2D ResUNet++-style model for multi-class segmentation."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(self.pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SqueezeExcitation(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.convs(x) + self.shortcut(x))


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = max(out_channels // 4, 1)

        def branch(kernel_size: int, dilation: int = 1):
            return nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size,
                          padding=dilation if kernel_size > 1 else 0,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branches = nn.ModuleList([branch(1), branch(3, 6), branch(3, 12)])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        pooled = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, attention_channels: int):
        super().__init__()
        self.skip_projection = nn.Conv2d(skip_channels, attention_channels, kernel_size=1, bias=False)
        self.gate_projection = nn.Conv2d(gate_channels, attention_channels, kernel_size=1, bias=False)
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        weights = self.attention(self.skip_projection(skip) + self.gate_projection(gate))
        return skip * weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.attention = AttentionGate(skip_channels, in_channels, out_channels)
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(skip, x)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNetPP(nn.Module):
    """ResUNet++ encoder-decoder returning raw segmentation logits."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        kernels = kwargs.get("kernels", 32)
        channels = [kernels, kernels * 2, kernels * 4, kernels * 8]

        self.encoder1 = ResidualBlock(in_dim, channels[0])
        self.encoder2 = ResidualBlock(channels[0], channels[1])
        self.encoder3 = ResidualBlock(channels[1], channels[2])
        self.encoder4 = ResidualBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ASPP(channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor:
        skip1 = self.encoder1(input)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        encoded = self.encoder4(self.pool(skip3))
        encoded = self.bottleneck(self.pool(encoded))
        decoded = self.decoder3(encoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        return self.head(F.interpolate(decoded, size=input.shape[-2:], mode="bilinear", align_corners=False))

    def init_weights(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)