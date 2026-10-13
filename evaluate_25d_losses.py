#!/usr/bin/env python3

"""Evaluate 2.5D checkpoints trained with different losses on one validation set."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from TwoPointFiveD import TwoPointFiveD
from dataset import SliceDataset
from losses import make_loss
from main import datasets_params, gt_transform, img_transform
from segthor_labels import decode_png
from utils import dice_coef, hd95_coef, iou_coef, probs2one_hot


LOSS_NAMES = ("cross_entropy", "dice", "dice_cross_entropy", "focal_tversky")


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in LOSS_NAMES or not path:
        raise argparse.ArgumentTypeError(
            f"checkpoint must have the form loss=PATH, with loss in {LOSS_NAMES}"
        )
    return name, Path(path)


def checkpoint_paths(root: Path, explicit: list[tuple[str, Path]]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {name: [] for name in LOSS_NAMES}
    for name, path in explicit:
        found[name].append(path)
    for name in LOSS_NAMES:
        if found[name]:
            continue
        candidates = sorted(
            path for path in (root / name).glob("**/bestweights.pt")
            if path.is_file()
        )
        if not candidates and (root / name / "bestweights.pt").is_file():
            candidates = [root / name / "bestweights.pt"]
        found[name] = candidates
    return found


def build_model(args, device: torch.device, classes: int) -> TwoPointFiveD:
    model = TwoPointFiveD(
        1,
        classes,
        kernels=args.kernels,
        factor=args.factor,
        context_slices=args.context_slices,
        backbone=getattr(args, "backbone", "unet"),
    ).to(device)
    return model


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model.load_state_dict(checkpoint.state_dict())
        return
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    model.load_state_dict(state_dict)


def evaluate(model, loader, loss_fn, device, classes):
    loss_sum = 0.0
    batches = 0
    dice_values, iou_values, hd95_values = [], [], []
    with torch.no_grad():
        for data in loader:
            images = data["images"].to(device)
            targets = data["gts"].to(device)
            probabilities = F.softmax(model(images), dim=1)
            predictions = probs2one_hot(probabilities)

            loss_sum += float(loss_fn(probabilities, targets))
            batches += 1
            dice_values.append(dice_coef(predictions, targets).cpu().numpy())
            iou_values.append(iou_coef(predictions, targets).cpu().numpy())
            hd95_values.append(hd95_coef(predictions, targets).cpu().numpy())

    if batches == 0:
        raise RuntimeError("Validation dataset is empty")
    metrics = {
        "loss": loss_sum / batches,
        "dice_per_class": np.concatenate(dice_values).mean(axis=0).tolist(),
        "iou_per_class": np.concatenate(iou_values).mean(axis=0).tolist(),
        "hd95_per_class": np.concatenate(hd95_values).mean(axis=0).tolist(),
    }
    metrics["dice_foreground_mean"] = float(np.mean(metrics["dice_per_class"][1:]))
    metrics["iou_foreground_mean"] = float(np.mean(metrics["iou_per_class"][1:]))
    metrics["hd95_foreground_mean"] = float(np.mean(metrics["hd95_per_class"][1:]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SEGTHOR_FIXED", choices=datasets_params)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/SEGTHOR_FIXED/25d"))
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, default=[],
                        help="Override discovery: loss=PATH. May be repeated.")
    parser.add_argument("--output", type=Path, default=Path("results/SEGTHOR_FIXED/25d/evaluation"))
    parser.add_argument("--backbone", choices=("unet", "vnet"), default="unet",
                        help="Backbone architecture for the 2.5D model.")
    parser.add_argument("--context-slices", type=int, default=3)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--mode", choices=("full", "partial"), default="full")
    parser.add_argument("--experiment", choices=("B0", "B5"), default="B0")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.context_slices < 1 or args.context_slices % 2 == 0:
        parser.error("--context-slices must be a positive odd number")
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    classes = datasets_params[args.dataset]["K"]
    idk = list(range(classes)) if args.mode == "full" else [0, 1, 3, 4]
    validation = SliceDataset(
        "val",
        Path("data") / args.dataset,
        img_transform=img_transform,
        gt_transform=lambda image: gt_transform(classes, image),
        debug=args.debug,
        experiment=args.experiment,
        context_slices=args.context_slices,
    )
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    discovered = checkpoint_paths(args.checkpoint_root, args.checkpoint)
    missing = [name for name, paths in discovered.items() if not paths]
    if missing:
        raise FileNotFoundError(
            "No checkpoints found for: " + ", ".join(missing)
            + ". Use --checkpoint loss=PATH or the expected checkpoint-root/loss/ layout."
        )

    rows = []
    for loss_name, paths in discovered.items():
        for checkpoint in paths:
            model = build_model(args, device, classes)
            load_weights(model, checkpoint, device)
            model.eval()
            metrics = evaluate(model, loader, make_loss(loss_name, idk=idk), device, classes)
            rows.append({"loss_name": loss_name, "checkpoint": str(checkpoint), **metrics})
            print(f"{loss_name:18s} {checkpoint}: "
                  f"Dice={metrics['dice_foreground_mean']:.4f}, "
                  f"IoU={metrics['iou_foreground_mean']:.4f}, "
                  f"HD95={metrics['hd95_foreground_mean']:.4f}, "
                  f"loss={metrics['loss']:.6f}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
