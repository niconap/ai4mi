#!/usr/bin/env python3
"""Create a side-by-side montage of predictions from arbitrary models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path))


def prediction_sets(root: Path) -> list[tuple[str, Path]]:
    """Find directories containing patient slice PNGs below a prediction root."""
    candidates = sorted({
        path.parent for path in root.rglob("Patient_*.png")
    })
    if not candidates:
        raise FileNotFoundError(f"No Patient_*.png predictions found below {root}")

    discovered = []
    for directory in candidates:
        relative = directory.relative_to(root)
        parts = relative.parts
        label_parts = parts[:-1] if parts and parts[-1] == "run_01" else parts
        label = "/".join(label_parts) if label_parts else root.name
        discovered.append((label, directory))
    return discovered


def slice_names(image_root: Path, patient: str, requested: list[int] | None) -> list[str]:
    available = sorted(image_root.glob(f"{patient}_*.png"))
    if not available:
        raise FileNotFoundError(f"No slices found for {patient} below {image_root}")
    if requested is None:
        positions = np.linspace(0, len(available) - 1, min(6, len(available)), dtype=int)
        return [available[position].name for position in positions]
    names = [f"{patient}_{index:04d}.png" for index in requested]
    missing = [name for name in names if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing requested slices: " + ", ".join(missing))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Sliced test root containing test/img.")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Root containing prediction PNGs, recursively.")
    parser.add_argument("--patient", required=True, help="For example Patient_41.")
    parser.add_argument("--slices", type=int, nargs="+",
                        help="Zero-based slice indices; defaults to six evenly spaced slices.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    image_root = args.images / "test" / "img"
    names = slice_names(image_root, args.patient, args.slices)
    configurations = prediction_sets(args.predictions)
    columns = ("CT",) + tuple(label for label, _ in configurations)
    figure, axes = plt.subplots(
        len(names), len(columns),
        figsize=(3.2 * len(columns), 3.2 * len(names)),
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10", args.num_classes)
    for column, title in enumerate(columns):
        axes[0, column].set_title(title)
    for row, name in enumerate(names):
        image = read_png(image_root / name)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=62, va="center")
        for column, (label, directory) in enumerate(configurations, start=1):
            prediction = read_png(directory / name)
            if prediction.shape != image.shape:
                raise ValueError(f"{name}: image shape {image.shape} != prediction shape {prediction.shape}")
            if not np.isin(prediction, np.arange(args.num_classes)).all():
                raise ValueError(f"{name}: unexpected classes {np.unique(prediction)}")
            axis = axes[row, column]
            axis.imshow(image, cmap="gray")
            axis.imshow(prediction, cmap=cmap, vmin=0, vmax=args.num_classes - 1,
                        alpha=args.alpha)
        for axis in axes[row]:
            axis.set_axis_off()
    figure.suptitle(f"{args.patient} — model predictions", fontsize=16)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {len(names)} slices to {args.output}")


if __name__ == "__main__":
    main()
