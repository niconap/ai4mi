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

from pathlib import Path
from typing import Callable, Union
from collections import defaultdict

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from augmentation import SliceAugmentation, bilateral_denoise


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 experiment="B0", context_slices=1):
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number")
        if augment:
            raise ValueError("Use experiment=B1 through B6 instead of augment=True")
        if subset != "train" and experiment not in ("B0", "B5", "B6"):
            raise ValueError("Augmentation is only allowed on the training split")
        self.transform_pair = SliceAugmentation(experiment if subset == "train" else "B0")
        self.denoising = experiment == "B5"
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.experiment: str = experiment
        self.context_slices = context_slices
        self.augmentation: bool = subset == "train" and experiment not in ("B0", "B5")
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        self._context_groups = defaultdict(list)
        for file_index, (image_path, _) in enumerate(self.files):
            stem_parts = image_path.stem.rsplit("_", 1)
            group = stem_parts[0] if len(stem_parts) == 2 and stem_parts[1].isdigit() else image_path.stem
            self._context_groups[group].append(file_index)
        self._context_group_by_index = {
            file_index: (group, position)
            for group, indices in self._context_groups.items()
            for position, file_index in enumerate(indices)
        }

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        group, center = self._context_group_by_index[index]
        group_indices = self._context_groups[group]
        radius = self.context_slices // 2
        context_indices = [group_indices[min(max(center + offset, 0), len(group_indices) - 1)]
                   for offset in range(-radius, radius + 1)]
        img = torch.cat([self.img_transform(Image.open(self.files[context_index][0]))
                 for context_index in context_indices], dim=0)
        # Deterministic preprocessing for B5
        if self.denoising:
            img = bilateral_denoise(img)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            img, gt = self.transform_pair(img, gt)
            data_dict["images"] = img

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
