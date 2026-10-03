"""2.5D segmentation models that use adjacent slices as input channels."""

from torch import Tensor, nn

from UNet import UNet


class TwoPointFiveD(nn.Module):
    """A 2D U-Net fed with a configurable axial slice context window."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        context_slices = kwargs.pop("context_slices", 3)
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number")
        self.context_slices = context_slices
        self.backbone = UNet(context_slices * in_dim, out_dim, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.context_slices:
            raise ValueError(
                f"Expected {self.context_slices} context channels, got {input.shape[1]}"
            )
        return self.backbone(input)

    def init_weights(self, *args, **kwargs):
        self.backbone.init_weights(*args, **kwargs)