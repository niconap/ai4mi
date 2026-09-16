#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("dice_tra", "dice_val", "loss_tra", "loss_val")


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
            arrays = [np.load(run_dir / f"{metric}.npy") for run_dir in run_dirs]
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
        summary[model] = model_summary

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Saved averaged metrics to {output_root}")


if __name__ == "__main__":
    main()