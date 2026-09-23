"""Cumulative, training-only B0–B4 augmentation for normalized slice tensors."""
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

EXPERIMENTS = ("B0", "B1", "B2", "B3", "B4")


def uniform(low, high):
    return torch.empty(()).uniform_(low, high).item()


class SliceAugmentation:
    def __init__(self, experiment="B0"):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        self.level = EXPERIMENTS.index(experiment)

    def __call__(self, image, target):
        if self.level == 0:
            return image, target
        height, width = image.shape[-2:]
        # One geometry draw shared by image and discrete class map.
        if torch.rand(()).item() < 0.8:
            angle = uniform(-10, 10)
            scale = uniform(0.9, 1.1)
            translation = [round(uniform(-0.05, 0.05) * width),
                           round(uniform(-0.05, 0.05) * height)]
            kwargs = dict(angle=angle, translate=translation, scale=scale,
                          shear=[0.0, 0.0], fill=0)
            image = TF.affine(image, interpolation=InterpolationMode.BILINEAR, **kwargs)
            labels = target.argmax(0, keepdim=True).float()
            labels = TF.affine(labels, interpolation=InterpolationMode.NEAREST, **kwargs)
            target = F.one_hot(labels[0].long(), target.shape[0]).permute(2, 0, 1).to(target.dtype)
        if self.level >= 2:
            if torch.rand(()).item() < 0.5:
                image = TF.adjust_contrast(image, uniform(0.85, 1.15))
            if torch.rand(()).item() < 0.5:
                image = TF.adjust_gamma(image, uniform(0.85, 1.15))
        if self.level >= 3 and torch.rand(()).item() < 0.3:
            image = (image + torch.randn_like(image) * uniform(0.0, 0.03)).clamp(0, 1)
        if self.level >= 4:
            if torch.rand(()).item() < 0.2:
                image = TF.gaussian_blur(image, [5, 5], [uniform(0.3, 0.8)] * 2)
            if torch.rand(()).item() < 0.2:
                factor = uniform(0.65, 0.9)
                small = [max(1, round(height * factor)), max(1, round(width * factor))]
                image = F.interpolate(image[None], size=small, mode="area")
                image = F.interpolate(image, size=(height, width), mode="bilinear",
                                      align_corners=False)[0]
        return image.clamp(0, 1), target
