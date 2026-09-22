#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def read_mask(path: Path) -> np.ndarray:
    values = read_image(path)
    return values / 63 if values.max() > 4 else values


def model_name(prediction_dir: Path, results: Path) -> str:
    relative_parts = prediction_dir.relative_to(results).parts
    for model in ("enet", "unet"):
        if model in relative_parts:
            return model
    return prediction_dir.parent.parent.name


def prediction_label(prediction_dir: Path, results: Path) -> str:
    relative_parts = prediction_dir.relative_to(results).parts
    best_epoch_index = relative_parts.index("best_epoch")
    return "/".join(relative_parts[max(0, best_epoch_index - 2):best_epoch_index])


def plot_average_metrics(args: argparse.Namespace) -> None:
    average_root = args.average_results
    metric_paths = {
        model: {
            "dice": average_root / model / "dice_val_mean.npy",
            "loss": average_root / model / "loss_val_mean.npy",
        }
        for model in ("enet", "unet")
    }
    available = {
        model: paths for model, paths in metric_paths.items()
        if all(path.is_file() for path in paths.values())
    }
    if not available:
        print(f"No averaged metrics found under {average_root}; skipped metric plot")
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, paths in available.items():
        dice = np.load(paths["dice"])
        loss = np.load(paths["loss"])
        axes[0].plot(np.mean(dice[:, :, 1:], axis=(1, 2)), label=model)
        axes[1].plot(np.mean(loss, axis=1), label=model)
    axes[0].set_title("Average validation Dice")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Foreground Dice")
    axes[1].set_title("Average validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    metrics_output = args.output.with_name(f"{args.output.stem}_metrics.png")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(metrics_output, dpi=150, bbox_inches="tight")
    print(f"Saved averaged metric curves to {metrics_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a montage of random SegTHOR slices and model predictions."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/SEGTHOR"))
    parser.add_argument("--subset", choices=["train", "val"], default="val")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("comparison.png"))
    parser.add_argument("--average-results", type=Path,
                        default=Path("results/SEGTHOR/averages"))
    args = parser.parse_args()

    image_dir = args.dataset / args.subset / "img"
    gt_dir = args.dataset / args.subset / "gt"
    prediction_dirs = sorted(args.results.rglob("best_epoch/val"))
    if not image_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset images or labels under {args.dataset}")
    if not prediction_dirs:
        raise FileNotFoundError(f"No */best_epoch/val directories found under {args.results}")

    prediction_groups = {}
    for prediction_dir in prediction_dirs:
        prediction_groups.setdefault(model_name(prediction_dir, args.results), []).append(prediction_dir)

    sources = [image_dir, gt_dir, *prediction_dirs]
    common_names = set(path.name for path in sources[0].glob("*.png"))
    for source in sources[1:]:
        common_names &= {path.name for path in source.glob("*.png")}
    if not common_names:
        raise FileNotFoundError("No slices are shared by the dataset and all model results")

    random.seed(args.seed)
    sample_names = random.sample(sorted(common_names), min(args.samples, len(common_names)))
    columns = [("image", image_dir), ("ground truth", gt_dir)]
    for prediction_dir in prediction_dirs:
        columns.append((prediction_label(prediction_dir, args.results), prediction_dir))
    for model, model_dirs in sorted(prediction_groups.items()):
        run_dirs = [path for path in model_dirs if "run_" in path.relative_to(args.results).parts]
        if run_dirs:
            columns.append((f"{model} average ({len(run_dirs)} runs)", run_dirs))

    figure, axes = plt.subplots(
        len(sample_names), len(columns),
        squeeze=False,
        figsize=(3 * len(columns), 3 * len(sample_names)),
    )
    for column, (title, _) in enumerate(columns):
        axes[0, column].set_title(title)

    for row, name in enumerate(sample_names):
        image = read_image(image_dir / name)
        for column, (_, source) in enumerate(columns):
            axis = axes[row, column]
            if column == 0:
                values = read_image(source / name)
                axis.imshow(values, cmap="gray")
            else:
                if isinstance(source, list):
                    values = np.mean([read_mask(path / name) for path in source], axis=0)
                else:
                    values = read_mask(source / name)
                axis.imshow(image, cmap="gray")
                axis.imshow(values, cmap="tab10", alpha=0.45, vmin=0, vmax=4)
            axis.set_axis_off()
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=55, va="center")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {len(sample_names)} samples to {args.output}")
    plot_average_metrics(args)


if __name__ == "__main__":
    main()