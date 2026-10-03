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
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
    for seg, name in zip(segs, names):
        save_path = ((root / name).with_suffix(".npy")
                 if len(seg.shape) == 3 else
                 (root / name).with_suffix(".png"))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(seg.shape) == 2:
            Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
        elif len(seg.shape) == 3:
            np.save(str(save_path), seg.detach().cpu().numpy())
        else:
            raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def meta_iou(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    union_size: Tensor = einsum(sum_str, [union(label, pred)]).type(torch.float32)

    ious: Tensor = (inter_size + smooth) / (union_size + smooth)

    return ious


iou_coef = partial(meta_iou, "bk...->bk")
iou_batch = partial(meta_iou, "bk...->k")  # used for 3d iou


def hausdorff95_single(pred: np.ndarray, label: np.ndarray,
                       voxelspacing: tuple[float, ...] | None = None,
                       max_dist: float | None = None,
                       percentile: float = 95.0) -> float:
    """Compute the 95th percentile Hausdorff Distance between two binary masks.

    Args:
        pred: Binary mask of predicted segmentation (2D or 3D bool/int array).
        label: Binary mask of ground truth segmentation (2D or 3D bool/int array).
        voxelspacing: Voxel spacing along each spatial dimension.
        max_dist: Maximum penalty distance if exactly one mask is empty. Defaults to the
                  spatial domain diagonal.
        percentile: Distance percentile to compute (default: 95.0).

    Returns:
        The percentile Hausdorff distance (0.0 if both masks are empty, max_dist if
        exactly one mask is empty).
    """
    pred_bool = np.asarray(pred, dtype=bool)
    label_bool = np.asarray(label, dtype=bool)

    pred_empty = not np.any(pred_bool)
    label_empty = not np.any(label_bool)

    if pred_empty and label_empty:
        return 0.0

    spatial_shape = pred_bool.shape
    if max_dist is None:
        if voxelspacing is not None:
            max_dist = float(np.sqrt(sum((s * sp) ** 2 for s, sp in zip(spatial_shape, voxelspacing))))
        else:
            max_dist = float(np.sqrt(sum(s ** 2 for s in spatial_shape)))

    if pred_empty or label_empty:
        return float(max_dist)

    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        footprint = np.ones((3,) * pred_bool.ndim, dtype=bool)
        pred_border = pred_bool ^ binary_erosion(pred_bool, structure=footprint)
        label_border = label_bool ^ binary_erosion(label_bool, structure=footprint)

        if not np.any(pred_border):
            pred_border = pred_bool
        if not np.any(label_border):
            label_border = label_bool

        dt_label = distance_transform_edt(~label_border, sampling=voxelspacing)
        dt_pred = distance_transform_edt(~pred_border, sampling=voxelspacing)

        d_pred_to_label = dt_label[pred_border]
        d_label_to_pred = dt_pred[label_border]
    except ImportError:
        def get_border(m):
            padded = np.pad(m, 1, mode='constant', constant_values=0)
            slices = [slice(1, -1)] * m.ndim
            eroded = np.ones_like(m, dtype=bool)
            for d in range(m.ndim):
                for s in (-1, 1):
                    neighbor_slice = list(slices)
                    neighbor_slice[d] = slice(1 + s, -1 + s if -1 + s != 0 else None)
                    eroded &= padded[tuple(neighbor_slice)]
            border = m ^ eroded
            return border if np.any(border) else m

        pred_border = get_border(pred_bool)
        label_border = get_border(label_bool)

        pts_pred = np.argwhere(pred_border).astype(float)
        pts_label = np.argwhere(label_border).astype(float)
        if voxelspacing is not None:
            spacing = np.array(voxelspacing, dtype=float)
            pts_pred *= spacing
            pts_label *= spacing

        d_pred_to_label = [float(np.min(np.linalg.norm(pts_label - p, axis=1))) for p in pts_pred]
        d_label_to_pred = [float(np.min(np.linalg.norm(pts_pred - l, axis=1))) for l in pts_label]

    all_dists = np.concatenate([d_pred_to_label, d_label_to_pred])
    return float(np.percentile(all_dists, percentile))


def hausdorff95(pred: Tensor | np.ndarray, label: Tensor | np.ndarray,
                voxelspacing: tuple[float, ...] | None = None,
                max_dist: float | None = None,
                percentile: float = 95.0) -> Tensor:
    """Compute HD95 for a batch of predictions and targets.

    Expects one-hot or binary tensors of shape (B, K, ...) or (K, ...).
    Returns a Tensor of shape (B, K) or (K,).
    """
    is_torch = isinstance(pred, Tensor)
    device = pred.device if is_torch else None

    if is_torch:
        pred_np = pred.detach().cpu().numpy().astype(bool)
        label_np = label.detach().cpu().numpy().astype(bool)
    else:
        pred_np = np.asarray(pred, dtype=bool)
        label_np = np.asarray(label, dtype=bool)

    assert pred_np.shape == label_np.shape, f"Shape mismatch: {pred_np.shape} vs {label_np.shape}"

    if pred_np.ndim >= 3:
        B, K = pred_np.shape[0], pred_np.shape[1]
        res = np.zeros((B, K), dtype=np.float32)
        for b in range(B):
            for k in range(K):
                res[b, k] = hausdorff95_single(pred_np[b, k], label_np[b, k],
                                               voxelspacing=voxelspacing,
                                               max_dist=max_dist,
                                               percentile=percentile)
    elif pred_np.ndim == 2:
        K = pred_np.shape[0]
        res = np.zeros(K, dtype=np.float32)
        for k in range(K):
            res[k] = hausdorff95_single(pred_np[k], label_np[k],
                                        voxelspacing=voxelspacing,
                                        max_dist=max_dist,
                                        percentile=percentile)
    else:
        res = np.float32(hausdorff95_single(pred_np, label_np,
                                            voxelspacing=voxelspacing,
                                            max_dist=max_dist,
                                            percentile=percentile))

    return torch.from_numpy(res).to(device) if is_torch else torch.from_numpy(np.atleast_1d(res))


hd95_coef = hausdorff95
hausdorff95_coef = hausdorff95

