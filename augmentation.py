import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

AUGMENTATION_LEVELS = {
    "B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6,
}
EXPERIMENTS = tuple(AUGMENTATION_LEVELS)


def bilateral_denoise(image):

    padded = F.pad(image[None], (2, 2, 2, 2), mode="replicate")[0]
    height, width = image.shape[-2:]
    weighted = torch.zeros_like(image)
    total = torch.zeros_like(image)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            neighbor = padded[:, 2 + dy:2 + dy + height, 2 + dx:2 + dx + width]
            weight = torch.exp(-0.5 * ((neighbor - image) / 0.02).square()
                               - 0.5 * (dx * dx + dy * dy))
            weighted += weight * neighbor
            total += weight
    return (weighted / total).clamp(0, 1)


def mild_elastic(image, target):
    """deformation capped at 1 pixel """
    
    height, width = image.shape[-2:]
    coarse = torch.empty((1, 2, 9, 9), device=image.device,
                         dtype=image.dtype).uniform_(-1, 1)
    displacement = F.interpolate(coarse, size=(height, width), mode="bicubic",
                                 align_corners=True)
    displacement /= displacement.square().sum(1).sqrt().amax().clamp_min(1.0)
    y, x = torch.meshgrid(torch.arange(height, device=image.device, dtype=image.dtype),
                          torch.arange(width, device=image.device, dtype=image.dtype),
                          indexing="ij")
    grid = torch.stack((2 * (x + 0.5) / width - 1,
                        2 * (y + 0.5) / height - 1), dim=-1)[None]
    grid[..., 0] += displacement[:, 0] * (2 / width)
    grid[..., 1] += displacement[:, 1] * (2 / height)
    image = F.grid_sample(image[None], grid, mode="bilinear",
                          padding_mode="border", align_corners=False)[0]
    labels = target.argmax(0, keepdim=True).to(image.dtype)
    labels = F.grid_sample(labels[None], grid, mode="nearest",
                           padding_mode="zeros", align_corners=False)[0, 0].long()
    target = F.one_hot(labels, target.shape[0]).permute(2, 0, 1).to(target.dtype)
    return image, target


def uniform(low, high):
    return torch.empty(()).uniform_(low, high).item()


class SliceAugmentation:
    def __init__(self, experiment="B0"):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        self.level = AUGMENTATION_LEVELS[experiment]

    def __call__(self, image, target):
        if self.level in (0, 5):
            return image, target
        height, width = image.shape[-2:]
        # nn interpolation for seg mask 
        # so affine transf don't create new classes 
        if self.level == 1 and torch.rand(()).item() < 0.8:
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
        if self.level == 6 and torch.rand(()).item() < 0.05:
            image, target = mild_elastic(image, target)
        if self.level == 2:
            if torch.rand(()).item() < 0.5:
                image = TF.adjust_contrast(image, uniform(0.85, 1.15))
            if torch.rand(()).item() < 0.5:
                image = TF.adjust_gamma(image, uniform(0.85, 1.15))
        if self.level == 3 and torch.rand(()).item() < 0.3:
            image = (image + torch.randn_like(image) * uniform(0.0, 0.03)).clamp(0, 1)
        if self.level == 4:
            if torch.rand(()).item() < 0.2:
                image = TF.gaussian_blur(image, [5, 5], [uniform(0.3, 0.8)] * 2)
            if torch.rand(()).item() < 0.2:
                factor = uniform(0.65, 0.9)
                small = [max(1, round(height * factor)), max(1, round(width * factor))]
                image = F.interpolate(image[None], size=small, mode="area")
                image = F.interpolate(image, size=(height, width), mode="bilinear",
                                      align_corners=False)[0]
        return image.clamp(0, 1), target
