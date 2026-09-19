#!/usr/bin/env python3

# Sampling strategies for the training slices.
#
# The loader draws every axial slice with equal probability by default, but a
# third of them contain no organ at all, and the classes differ by two orders of
# magnitude in size. These samplers redistribute the per-epoch budget without
# touching the dataset, the loss, or the number of iterations per epoch.

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

Files = list[tuple[Path, Path]]


def slice_presence(files: Files, num_classes: int = 5) -> np.ndarray:
    """Whether each foreground label occurs in each slice, shape (N, K - 1)."""
    presence: np.ndarray = np.zeros((len(files), num_classes - 1), dtype=bool)

    for i, (image_path, gt_path) in enumerate(files):
        if image_path.stem != gt_path.stem:
            raise ValueError(f"Mismatched pair: {image_path} vs {gt_path}")

        with Image.open(gt_path) as image:
            # The slicing script stores the classes as {0, 63, ..., 63 * (K - 1)}
            labels: np.ndarray = np.asarray(image, dtype=np.uint8) // 63
            counts: np.ndarray = np.bincount(labels.ravel(), minlength=num_classes)
            presence[i] = counts[1:num_classes] > 0

    return presence


def _weighted_sampler(weights: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Draw one epoch of indices, so every strategy gets the same iteration budget."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights),
                                 replacement=True,
                                 generator=generator)


def label_mixture_weights(presence: np.ndarray,
                          mixture: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Mix uniform slice probabilities with label-balanced probabilities."""
    if not 0.0 <= mixture <= 1.0:
        raise ValueError("mixture must be between 0 and 1")

    active: np.ndarray = presence.any(axis=0)
    if not active.any():
        raise ValueError("No foreground labels found")

    observed: np.ndarray = presence[:, active]
    number_of_slices, number_of_labels = observed.shape

    # Every active label receives the same share (mixture / number_of_labels),
    # spread evenly over the slices in which it occurs.
    weights: np.ndarray = ((1.0 - mixture) / number_of_slices
                           + (mixture / number_of_labels)
                           * (observed / observed.sum(axis=0)).sum(axis=1))

    return weights, active


def make_label_sampler(files: Files, seed: int, mixture: float = 0.5
                       ) -> tuple[WeightedRandomSampler, np.ndarray, np.ndarray]:
    presence: np.ndarray = slice_presence(files)
    weights, active = label_mixture_weights(presence, mixture)

    return _weighted_sampler(weights, seed), presence, active


def make_foreground_sampler(files: Files, seed: int, foreground_fraction: float = 0.75
                            ) -> tuple[WeightedRandomSampler, np.ndarray]:
    """Sample a fixed proportion of foreground and background-only slices."""
    if not 0.0 < foreground_fraction < 1.0:
        raise ValueError("foreground_fraction must be between 0 and 1")

    presence: np.ndarray = slice_presence(files)
    has_foreground: np.ndarray = presence.any(axis=1)

    if has_foreground.all() or not has_foreground.any():
        raise ValueError("Foreground sampling requires both foreground and background-only slices")

    weights: np.ndarray = np.where(has_foreground,
                                   foreground_fraction / has_foreground.sum(),
                                   (1.0 - foreground_fraction) / (~has_foreground).sum())

    return _weighted_sampler(weights, seed), presence


def make_patient_sampler(files: Files, seed: int
                         ) -> tuple[WeightedRandomSampler, np.ndarray, np.ndarray, np.ndarray]:
    """Give every patient equal total sampling probability."""
    patient_ids: np.ndarray = np.asarray([image_path.stem.rsplit("_", 1)[0]
                                          for image_path, _ in files])
    patients, inverse, counts = np.unique(patient_ids,
                                          return_inverse=True,
                                          return_counts=True)

    return _weighted_sampler(1.0 / counts[inverse], seed), patient_ids, patients, counts
