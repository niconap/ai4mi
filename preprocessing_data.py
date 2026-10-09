import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from augmentation import SliceAugmentation, bilateral_denoise
from preprocess_ct import restore_probabilities
from segthor_labels import decode_png


class FloatSliceDataset(Dataset):
    def __init__(self, subset, root_dir, experiment='B0', normalization='window', context=1, debug=False):
        if subset not in ('train', 'val', 'test') or context not in (1, 3):
            raise ValueError('Invalid split or context.')
        if normalization not in ('window', 'zscore'):
            raise ValueError('Invalid intensity normalization.')
        self.root = Path(root_dir)
        self.plan = json.loads((self.root / 'preprocessing.json').read_text())
        self.samples = sorted(self.plan['samples'][subset], key=lambda r: r['stem'])
        if debug:
            self.samples = self.samples[:10]
        if not self.samples:
            raise ValueError(f'No samples for {subset}.')
        self.subset, self.context, self.normalization = subset, context, normalization
        self.transform_pair = SliceAugmentation(experiment if subset == 'train' else 'B0')
        self.denoising = experiment == 'B5'
        self.stochastic = subset == 'train'
        self.channels_per_slice = len(self.plan['intensity_statistics']['mean'])
        self.in_channels = self.channels_per_slice * context
        stats = self.plan['intensity_statistics']
        self.mean = torch.tensor(stats['mean'] * context, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(stats['std'] * context, dtype=torch.float32)[:, None, None]
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid frozen training statistics.')
        self.lookup = {r['stem']: r for r in self.plan['samples'][subset]}
        self.neighbors = {}
        for patient, record in self.plan['patients'].items():
            if record['split'] != subset:
                continue
            indices = record['slice_order']
            available = set(indices)
            direction = -1 if record['orientation'][2][1] < 0 else 1
            for z in indices:
                self.neighbors[f'{patient}_{z:04d}'] = [
                    f'{patient}_{z + d * direction if z + d * direction in available else z:04d}'
                    for d in (-1, 0, 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]; stem = sample['stem']
        stems = [stem] if self.context == 1 else self.neighbors[stem]
        # Keep full slices for validation/test; only training uses random 256px patches.
        arrays = [np.load(self.root / self.subset / 'img' / f'{s}.npy', allow_pickle=False) for s in stems]
        image = torch.from_numpy(np.concatenate(arrays, axis=0))
        if image.ndim != 3 or image.shape[0] != self.in_channels:
            raise ValueError(f'{stem}: unexpected CT channel shape {tuple(image.shape)}.')
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f'{stem}: expected finite windowed floats in [0,1].')
        target = None
        if self.subset != 'test':
            with Image.open(self.root / self.subset / 'gt' / f'{stem}.png') as im:
                labels = torch.from_numpy(decode_png(np.asarray(im)).astype(np.int64))
            if labels.shape != image.shape[-2:]:
                raise ValueError(f'{stem}: image/mask dimensions differ.')
            target = F.one_hot(labels, 5).permute(2, 0, 1)
        if self.denoising:
            image = bilateral_denoise(image)
        if self.stochastic:
            h, w = image.shape[-2:]
            pad = (0, max(0, 256-w), 0, max(0, 256-h))
            image = F.pad(image, pad)
            labels = F.pad(target.argmax(0), pad)
            h, w = image.shape[-2:]
            y = int(torch.randint(h - 256 + 1, ())) if h > 256 else 0
            x = int(torch.randint(w - 256 + 1, ())) if w > 256 else 0
            image = image[:, y:y+256, x:x+256]
            target = F.one_hot(labels[y:y+256, x:x+256], 5).permute(2, 0, 1)
            image, target = self.transform_pair(image, target)
        if self.normalization == 'zscore':
            image = (image - self.mean) / self.std
        result = {'images': image, 'stems': stem}
        if target is not None:
            result['gts'] = target
            if self.subset == 'val':
                with Image.open(self.root / self.subset / 'reference_gt' / f'{stem}.png') as im:
                    reference = torch.from_numpy(decode_png(np.asarray(im)).astype(np.int64))
                result['metric_gts'] = F.one_hot(reference, 5).permute(2, 0, 1)
        return result

    def restore(self, probabilities, stem, reference_shape):
        record = self.plan['patients'][self.lookup[stem]['patient']]
        result = restore_probabilities(probabilities.detach().cpu().numpy(), record, reference_shape)
        return torch.from_numpy(result).to(probabilities.device)

    @property
    def padding_value(self):
        return -self.mean / self.std if self.normalization == 'zscore' else torch.zeros_like(self.mean)


def organ_sampler(dataset, probability=.5):
    if isinstance(dataset, FloatSliceDataset):
        if dataset.subset != 'train':
            raise ValueError('Organ-aware sampling is training-only.')
        positive = np.array([bool({1, 3} & set(s['classes'])) for s in dataset.samples])
    else:
        positive = []
        for _, path in dataset.files:
            with Image.open(path) as im:
                positive.append(bool(np.isin(decode_png(np.asarray(im)), [1, 3]).any()))
        positive = np.array(positive)
    weights = np.full(len(positive), (1-probability) / len(positive))
    weights[positive] += probability / positive.sum()
    return WeightedRandomSampler(torch.tensor(weights), len(weights), replacement=True)


def tiled_logits(net, images, padding_value=None, patch_size=256):
    _, channels, h, w = images.shape
    hp, wp = max(h, patch_size), max(w, patch_size)
    fill = images.new_zeros(channels, 1, 1) if padding_value is None else padding_value.to(images)
    padded = fill[None].expand(1, channels, hp, wp).clone()
    padded[:, :, :h, :w] = images
    def starts(n):
        return sorted(set(list(range(0, n-patch_size+1, patch_size//2)) + [n-patch_size]))
    total = None; count = images.new_zeros((1, 1, hp, wp))
    for y in starts(hp):
        for x in starts(wp):
            logits = net(padded[:, :, y:y+patch_size, x:x+patch_size])
            if total is None:
                total = logits.new_zeros((1, logits.shape[1], hp, wp))
            total[:, :, y:y+patch_size, x:x+patch_size] += logits
            count[:, :, y:y+patch_size, x:x+patch_size] += 1
    return (total / count)[:, :, :h, :w]
