"""A compact 3D U-Net for patient-volume segmentation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UNet3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        kernels = kwargs.get("kernels", 8)
        channels = [kernels, kernels * 2, kernels * 4, kernels * 8]
        self.enc1 = DoubleConv3D(in_dim, channels[0])
        self.enc2 = DoubleConv3D(channels[0], channels[1])
        self.enc3 = DoubleConv3D(channels[1], channels[2])
        self.bottleneck = DoubleConv3D(channels[2], channels[3])
        self.pool = nn.MaxPool3d(2)
        self.up3 = UpConv3D(channels[3], channels[2], channels[2])
        self.up2 = UpConv3D(channels[2], channels[1], channels[1])
        self.up1 = UpConv3D(channels[1], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], out_dim, kernel_size=1)
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor:
        skip1 = self.enc1(input)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        encoded = self.bottleneck(self.pool(skip3))
        decoded = self.up3(encoded, skip3)
        decoded = self.up2(decoded, skip2)
        decoded = self.up1(decoded, skip1)
        return self.head(decoded)

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         nn.Conv3d(in_channels, out_channels, 1, bias=False))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(self.sequence(input) + self.shortcut(input))


class UpConv3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, 2, stride=2)
        self.conv = DoubleConv3D(skip_channels * 2, out_channels)

    def forward(self, input: Tensor, skip: Tensor) -> Tensor:
        input = self.up(input)
        differences = [skip.size(axis) - input.size(axis) for axis in (2, 3, 4)]
        padding = []
        for difference in reversed(differences):
            padding.extend((difference // 2, difference - difference // 2))
        input = F.pad(input, padding)
        return self.conv(torch.cat([skip, input], dim=1))


def random_weights_init(module):
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)