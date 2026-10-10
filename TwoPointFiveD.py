"""2.5D segmentation models that use adjacent slices as input channels."""

from typing import Any

from torch import Tensor, nn

from UNet import UNet
from VNet import VNet


class TwoPointFiveD(nn.Module):
    """A 2.5D segmentation model fed with a configurable axial slice context window.

    By default, uses 2D U-Net as the backbone. A different backbone (e.g. 'vnet'
    or a custom nn.Module class) can be specified via the backbone keyword argument.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any):
        super().__init__()
        context_slices = kwargs.pop("context_slices", 3)
        backbone_type = kwargs.pop("backbone", "unet")
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number")
        self.context_slices = context_slices

        if isinstance(backbone_type, str):
            if backbone_type.lower() == "unet":
                backbone_cls = UNet
            elif backbone_type.lower() == "vnet":
                backbone_cls = VNet
            else:
                raise ValueError(
                    f"Unknown backbone: {backbone_type}. Supported backbones: 'unet', 'vnet'"
                )
        elif isinstance(backbone_type, type) and issubclass(backbone_type, nn.Module):
            backbone_cls = backbone_type
        else:
            raise TypeError(f"Unsupported backbone type: {type(backbone_type)}")

        self.backbone = backbone_cls(context_slices * in_dim, out_dim, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.context_slices:
            raise ValueError(
                f"Expected {self.context_slices} context channels, got {input.shape[1]}"
            )
        return self.backbone(input)

    def init_weights(self, *args: Any, **kwargs: Any):
        self.backbone.init_weights(*args, **kwargs)


class TwoPointFiveDVNet(nn.Module):
    """A 2.5D model using the 2D V-Net architecture as its backbone."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any):
        super().__init__()
        context_slices = kwargs.pop("context_slices", 3)
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number")
        self.context_slices = context_slices
        self.backbone = VNet(context_slices * in_dim, out_dim, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.context_slices:
            raise ValueError(
                f"Expected {self.context_slices} context channels, got {input.shape[1]}"
            )
        return self.backbone(input)

    def init_weights(self, *args: Any, **kwargs: Any):
        self.backbone.init_weights(*args, **kwargs)


VNet25D = TwoPointFiveDVNet