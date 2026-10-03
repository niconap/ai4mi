#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "dice_tra", "dice_val",
    "iou_tra", "iou_val",
    "hd95_tra", "hd95_val",
    "hausdorff95_tra", "hausdorff95_val",
    "loss_tra", "loss_val",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average metrics from repeated ENet and UNet runs."
    )
    parser.add_argument("--results", type=Path, default=Path("results/SEGTHOR"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["enet", "unet"])
    args = parser.parse_args()

    output_root = args.output or args.results / "averages"
    summary = {}
    for model in args.models:
        run_dirs = [args.results / model / f"run_{run:02d}" for run in range(1, args.runs + 1)]
        missing = [path for path in run_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing {model} run directories: {', '.join(map(str, missing))}"
            )

        model_output = output_root / model
        model_output.mkdir(parents=True, exist_ok=True)
        model_summary = {"runs": [str(path) for path in run_dirs]}
        for metric in METRICS:
            metric_files = [run_dir / f"{metric}.npy" for run_dir in run_dirs]
            if not all(p.is_file() for p in metric_files):
                continue
            arrays = [np.load(p) for p in metric_files]
            shapes = {array.shape for array in arrays}
            if len(shapes) != 1:
                raise ValueError(f"Inconsistent {metric} shapes for {model}: {shapes}")
            values = np.stack(arrays)
            np.save(model_output / f"{metric}_mean.npy", values.mean(axis=0))
            np.save(model_output / f"{metric}_std.npy", values.std(axis=0))

        dice_values = np.stack([
            np.load(run_dir / "dice_val.npy") for run_dir in run_dirs
        ])
        score_by_epoch = dice_values[:, :, :, 1:].mean(axis=(0, 2, 3))
        best_epoch = int(np.argmax(score_by_epoch))
        model_summary["best_mean_validation_dice"] = float(score_by_epoch[best_epoch])
        model_summary["best_epoch"] = best_epoch
        model_summary["validation_dice_by_epoch"] = score_by_epoch.tolist()

        if all((run_dir / "iou_val.npy").is_file() for run_dir in run_dirs):
            iou_values = np.stack([
                np.load(run_dir / "iou_val.npy") for run_dir in run_dirs
            ])
            iou_by_epoch = iou_values[:, :, :, 1:].mean(axis=(0, 2, 3))
            model_summary["best_mean_validation_iou"] = float(iou_by_epoch[best_epoch])
            model_summary["validation_iou_by_epoch"] = iou_by_epoch.tolist()

        hd95_file = "hd95_val.npy" if all((run_dir / "hd95_val.npy").is_file() for run_dir in run_dirs) else (
            "hausdorff95_val.npy" if all((run_dir / "hausdorff95_val.npy").is_file() for run_dir in run_dirs) else None
        )
        if hd95_file is not None:
            hd95_values = np.stack([
                np.load(run_dir / hd95_file) for run_dir in run_dirs
            ])
            hd95_by_epoch = hd95_values[:, :, :, 1:].mean(axis=(0, 2, 3))
            model_summary["mean_validation_hd95_at_best_epoch"] = float(hd95_by_epoch[best_epoch])
            model_summary["best_mean_validation_hd95"] = float(np.min(hd95_by_epoch))
            model_summary["validation_hd95_by_epoch"] = hd95_by_epoch.tolist()

        summary[model] = model_summary

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Saved averaged metrics to {output_root}")


if __name__ == "__main__":
    main()