#!/usr/bin/env python3

from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import make_volume_dataset


def read_labels(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) // 63 


class VoxelPatchDataset(Dataset):
    def __init__(self, root_dir, img_transform, gt_transform, strategy,
                 patch_size=(32, 128, 128), patches_per_volume=16,
                 foreground_fraction=0.75, debug=False):
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.strategy = strategy
        self.patch_size = tuple(patch_size)
        self.patches_per_volume = patches_per_volume
        self.foreground_fraction = foreground_fraction

        self.volumes = make_volume_dataset(root_dir, 'train')
        if debug:
            self.volumes = self.volumes[:2]

        # Voxels of each class in each slice: enough to draw a centre without
        # keeping the volumes in memory.
        self.counts = [np.stack([np.bincount(read_labels(path).ravel(), minlength=5)
                                 for path in gt_paths])
                       for _, _, gt_paths in self.volumes]
        print(f">> Created train patch dataset: {len(self.volumes)} patients "
              f"x {patches_per_volume} {strategy} patches of {self.patch_size}")

    def __len__(self):
        return len(self.volumes) * self.patches_per_volume

    def sample_center(self, counts: np.ndarray, gt_paths) -> tuple[int, int, int]:
        """Pick the classes a centre may have, then one voxel of them uniformly."""
        if self.strategy == 'uniform':
            classes = list(range(counts.shape[1]))
        else:
            organs = np.flatnonzero(counts[:, 1:].sum(axis=0)) + 1
            if len(organs) == 0 or torch.rand(()) >= self.foreground_fraction:
                classes = [0]
            elif self.strategy == 'foreground':
                classes = organs.tolist()
            else:
                classes = [int(organs[torch.randint(len(organs), ())])]

        per_slice = torch.as_tensor(counts[:, classes].sum(axis=1), dtype=torch.double)
        z = int(torch.multinomial(per_slice, 1))
        voxels = np.argwhere(np.isin(read_labels(gt_paths[z]), classes))
        y, x = voxels[torch.randint(len(voxels), ())]
        return z, int(y), int(x)

    def __getitem__(self, index):
        patient = index // self.patches_per_volume
        stem, img_paths, gt_paths = self.volumes[patient]
        counts = self.counts[patient]

        center = self.sample_center(counts, gt_paths)
        with Image.open(gt_paths[0]) as image:
            shape = (len(gt_paths), image.height, image.width)
        # Centre the patch on the voxel, shifted to stay inside the scan.
        d, h, w = (min(max(c - p // 2, 0), n - p)
                   for c, p, n in zip(center, self.patch_size, shape))
        D, H, W = self.patch_size
        box = (w, h, w + W, h + H)

        images = torch.stack([self.img_transform(Image.open(path).crop(box))
                              for path in img_paths[d:d + D]], dim=1)
        gts = torch.stack([self.gt_transform(Image.open(path).crop(box))
                           for path in gt_paths[d:d + D]], dim=1)
        return {"images": images, "gts": gts, "stems": stem}


@torch.no_grad()
def sliding_window_probs(net, images, patch_size, device, overlap: float = 0.5):
    """Softmax of a whole volume [1, C, D, H, W], averaged over overlapping patches."""
    shape = images.shape[2:]
    starts = [sorted({*range(0, n - p, max(1, int(p * (1 - overlap)))), n - p})
              for n, p in zip(shape, patch_size)]

    probs = None
    hits = torch.zeros(shape)
    for d, h, w in product(*starts):
        window = (..., slice(d, d + patch_size[0]), slice(h, h + patch_size[1]), slice(w, w + patch_size[2]))
        patch_probs = net(images[window].to(device)).softmax(dim=1).cpu()
        if probs is None:
            probs = torch.zeros((1, patch_probs.shape[1], *shape))
        probs[window] += patch_probs
        hits[window] += 1

    return probs / hits
