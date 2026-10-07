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
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 3D Evaluation Metrics: Relative Volume Error and Normalised Surface Distance
# ---------------------------------------------------------------------------

def relative_volume_error(pred: Tensor | np.ndarray,
                          target: Tensor | np.ndarray | None = None,
                          *,
                          label: Tensor | np.ndarray | None = None,
                          voxelspacing: tuple[float, ...] | None = None,
                          smooth: float = 1e-8,
                          absolute: bool = True,
                          empty_target_penalty: float | None = 1.0) -> Tensor:
    """Compute Relative Volume Error (RVE) for 3D predictions and targets.

    Assumes batched 3D tensors:
    - Shape (B, K, D, H, W): returns (B, K)
    - Shape (B, D, H, W):    returns (B,)
    - Shape (K, D, H, W):    returns (K,)
    - Shape (D, H, W):       returns scalar tensor

    Args:
        pred: Predicted binary, one-hot, or probability tensor/array.
        target: Ground truth binary or one-hot tensor/array. Can also be passed as `label`.
        label: Alias for `target`.
        voxelspacing: Physical voxel spacing (dx, dy, dz). When given, voxel volume
                      (dx * dy * dz) scales the sum.
        smooth: Small constant to avoid division by zero when empty_target_penalty is None.
        absolute: If True, computes |V_pred - V_target| / V_target.
                  If False, computes (V_pred - V_target) / V_target.
        empty_target_penalty: Value assigned when target is empty (V_target == 0) and
                              pred is non-empty (V_pred > 0). Defaults to 1.0. If None,
                              uses standard division with smooth.

    Returns:
        Tensor containing RVE values matching the batch/class dimensions.
    """
    if target is None:
        if label is not None:
            target = label
        elif isinstance(pred, (tuple, list)) and len(pred) == 2:
            pred, target = pred[0], pred[1]
        else:
            raise ValueError("target (or label) must be provided")

    is_torch = isinstance(pred, Tensor)
    device = pred.device if is_torch else None

    if is_torch:
        pred_t = pred.float()
        target_t = target.float() if isinstance(target, Tensor) else torch.as_tensor(target, dtype=torch.float32, device=device)
    else:
        pred_t = torch.as_tensor(pred, dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.float32)

    assert pred_t.shape == target_t.shape, f"Shape mismatch: {pred_t.shape} vs {target_t.shape}"

    if pred_t.ndim >= 3:
        spatial_dims = (-3, -2, -1)
        v_pred = pred_t.sum(dim=spatial_dims)
        v_target = target_t.sum(dim=spatial_dims)
    else:
        v_pred = pred_t.sum()
        v_target = target_t.sum()

    if voxelspacing is not None:
        voxel_vol = float(np.prod(voxelspacing))
        v_pred = v_pred * voxel_vol
        v_target = v_target * voxel_vol

    diff = (v_pred - v_target).abs() if absolute else (v_pred - v_target)

    both_empty = (v_target == 0) & (v_pred == 0)
    target_empty = (v_target == 0) & (v_pred > 0)

    if empty_target_penalty is not None:
        safe_denom = torch.where(v_target > 0, v_target, torch.ones_like(v_target))
        res = diff / safe_denom
        res = torch.where(target_empty, torch.full_like(res, empty_target_penalty), res)
        res = torch.where(both_empty, torch.zeros_like(res), res)
    else:
        res = diff / (v_target + smooth)
        res = torch.where(both_empty, torch.zeros_like(res), res)

    return res.to(device) if is_torch else res


def rve_batch(pred: Tensor | np.ndarray,
              target: Tensor | np.ndarray | None = None,
              **kwargs) -> Tensor:
    """Compute Relative Volume Error aggregated across the entire batch per class.

    For input (B, K, D, H, W), sums over both batch and spatial dimensions,
    returning shape (K,).
    """
    if target is None and isinstance(pred, (tuple, list)) and len(pred) == 2:
        pred, target = pred[0], pred[1]

    is_torch = isinstance(pred, Tensor)
    device = pred.device if is_torch else None
    if is_torch:
        pred_t = pred.float()
        target_t = target.float() if isinstance(target, Tensor) else torch.as_tensor(target, dtype=torch.float32, device=device)
    else:
        pred_t = torch.as_tensor(pred, dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.float32)

    if pred_t.ndim >= 5:
        spatial_and_batch = (0, -3, -2, -1)
        v_pred = pred_t.sum(dim=spatial_and_batch)
        v_target = target_t.sum(dim=spatial_and_batch)
        absolute = kwargs.get("absolute", True)
        diff = (v_pred - v_target).abs() if absolute else (v_pred - v_target)
        both_empty = (v_target == 0) & (v_pred == 0)
        target_empty = (v_target == 0) & (v_pred > 0)
        penalty = kwargs.get("empty_target_penalty", 1.0)
        if penalty is not None:
            safe_denom = torch.where(v_target > 0, v_target, torch.ones_like(v_target))
            res = diff / safe_denom
            res = torch.where(target_empty, torch.full_like(res, penalty), res)
            res = torch.where(both_empty, torch.zeros_like(res), res)
        else:
            smooth = kwargs.get("smooth", 1e-8)
            res = diff / (v_target + smooth)
            res = torch.where(both_empty, torch.zeros_like(res), res)
        return res.to(device) if is_torch else res
    return relative_volume_error(pred_t, target_t, **kwargs)



rve = relative_volume_error
rve_coef = relative_volume_error
relative_volume_difference = relative_volume_error
rvd = relative_volume_error


def normalized_surface_distance_single(pred: np.ndarray,
                                       label: np.ndarray,
                                       tau: float = 1.0,
                                       voxelspacing: tuple[float, ...] | None = None) -> float:
    """Compute the Normalized Surface Distance (NSD) between two 3D binary masks.

    Also known as Normalized Surface Dice or Surface Dice at tolerance tau.

    Args:
        pred: Binary mask of predicted segmentation (3D or 2D bool/int array).
        label: Binary mask of ground truth segmentation (3D or 2D bool/int array).
        tau: Tolerance distance threshold (in physical units if voxelspacing is given,
             otherwise in voxels). Default: 1.0.
        voxelspacing: Voxel spacing along each spatial dimension.

    Returns:
        The NSD score in [0.0, 1.0] (1.0 if both masks are empty, 0.0 if exactly one
        is empty).
    """
    pred_bool = np.asarray(pred, dtype=bool)
    label_bool = np.asarray(label, dtype=bool)

    pred_empty = not np.any(pred_bool)
    label_empty = not np.any(label_bool)

    if pred_empty and label_empty:
        return 1.0
    if pred_empty or label_empty:
        return 0.0

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

    pred_within = np.sum(np.asarray(d_pred_to_label) <= tau)
    label_within = np.sum(np.asarray(d_label_to_pred) <= tau)
    total_pts = len(d_pred_to_label) + len(d_label_to_pred)
    if total_pts == 0:
        return 1.0
    return float((pred_within + label_within) / total_pts)


def normalized_surface_distance(pred: Tensor | np.ndarray,
                                target: Tensor | np.ndarray | None = None,
                                *,
                                label: Tensor | np.ndarray | None = None,
                                tau: float = 1.0,
                                tolerance: float | None = None,
                                voxelspacing: tuple[float, ...] | None = None) -> Tensor:
    """Compute Normalized Surface Distance (NSD) for a batch of 3D predictions and targets.

    Also aliased as `normaliser_phase_distance`, `normalised_phase_distance`,
    `normalized_phase_distance`, `normalised_surface_distance`, and `nsd_coef`.

    Expects batched 3D tensors:
    - Shape (B, K, D, H, W): returns (B, K)
    - Shape (B, D, H, W):    returns (B,)
    - Shape (K, D, H, W):    returns (K,)
    - Shape (D, H, W):       returns scalar tensor

    Args:
        pred: Predicted one-hot, binary, or probability tensor of shape (B, K, D, H, W)
              or (B, D, H, W).
        target: Target one-hot or binary tensor matching pred. Can also be passed as `label`.
        label: Alias for `target`.
        tau: Tolerance distance (default: 1.0).
        tolerance: Alias for `tau`.
        voxelspacing: Physical voxel spacing tuple (dx, dy, dz) or None.

    Returns:
        Tensor containing NSD values in [0.0, 1.0].
    """
    if target is None:
        if label is not None:
            target = label
        elif isinstance(pred, (tuple, list)) and len(pred) == 2:
            pred, target = pred[0], pred[1]
        else:
            raise ValueError("target (or label) must be provided")

    if tolerance is not None:
        tau = tolerance

    is_torch = isinstance(pred, Tensor)
    device = pred.device if is_torch else None

    if is_torch:
        if pred.is_floating_point():
            pred_np = (pred.detach().cpu().numpy() > 0.5)
        else:
            pred_np = pred.detach().cpu().numpy().astype(bool)

        if isinstance(target, Tensor):
            if target.is_floating_point():
                label_np = (target.detach().cpu().numpy() > 0.5)
            else:
                label_np = target.detach().cpu().numpy().astype(bool)
        else:
            label_np = np.asarray(target, dtype=bool)
    else:
        pred_np = np.asarray(pred)
        if np.issubdtype(pred_np.dtype, np.floating):
            pred_np = (pred_np > 0.5)
        else:
            pred_np = pred_np.astype(bool)

        label_np = np.asarray(target)
        if np.issubdtype(label_np.dtype, np.floating):
            label_np = (label_np > 0.5)
        else:
            label_np = label_np.astype(bool)

    assert pred_np.shape == label_np.shape, f"Shape mismatch: {pred_np.shape} vs {label_np.shape}"

    if pred_np.ndim >= 5:
        # Batched multiclass 3D: (B, K, D, H, W)
        B, K = pred_np.shape[0], pred_np.shape[1]
        res = np.zeros((B, K), dtype=np.float32)
        for b in range(B):
            for k in range(K):
                res[b, k] = normalized_surface_distance_single(pred_np[b, k], label_np[b, k],
                                                               tau=tau,
                                                               voxelspacing=voxelspacing)
    elif pred_np.ndim == 4:
        dim0 = pred_np.shape[0]
        res = np.zeros(dim0, dtype=np.float32)
        for i in range(dim0):
            res[i] = normalized_surface_distance_single(pred_np[i], label_np[i],
                                                        tau=tau,
                                                        voxelspacing=voxelspacing)
    elif pred_np.ndim == 3:
        res = np.float32(normalized_surface_distance_single(pred_np, label_np,
                                                            tau=tau,
                                                            voxelspacing=voxelspacing))
    else:
        res = np.float32(normalized_surface_distance_single(pred_np, label_np,
                                                            tau=tau,
                                                            voxelspacing=voxelspacing))

    return torch.from_numpy(res).to(device) if is_torch else torch.from_numpy(np.atleast_1d(res))


def nsd_batch(pred: Tensor | np.ndarray,
              target: Tensor | np.ndarray | None = None,
              **kwargs) -> Tensor:
    """Compute average NSD across the batch per class, returning shape (K,)."""
    val = normalized_surface_distance(pred, target, **kwargs)
    if val.ndim >= 2:
        return val.mean(dim=0) if isinstance(val, Tensor) else val.mean(axis=0)
    return val



# Aliases for Normaliser / Normalised / Normalized Phase Distance and Surface Distance
normaliser_phase_distance = normalized_surface_distance
normalised_phase_distance = normalized_surface_distance
normalized_phase_distance = normalized_surface_distance
normaliser_phase_distance_single = normalized_surface_distance_single
normalised_phase_distance_single = normalized_surface_distance_single
normalized_phase_distance_single = normalized_surface_distance_single
normaliser_phase_distance_batch = nsd_batch
normalised_phase_distance_batch = nsd_batch
normalized_phase_distance_batch = nsd_batch

normalised_surface_distance = normalized_surface_distance
nsd = normalized_surface_distance
nsd_coef = normalized_surface_distance


