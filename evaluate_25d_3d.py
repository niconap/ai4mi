#!/usr/bin/env python3
"""Evaluate a 2.5D checkpoint on reconstructed 3D validation volumes."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from TwoPointFiveD import TwoPointFiveD
from dataset import SliceDataset
from main import datasets_params, gt_transform, img_transform
from utils import normalized_surface_distance_single, relative_volume_error


CLASS_NAMES = ("Background", "Esophagus", "Heart", "Trachea", "Aorta")
PATIENT_SLICE = re.compile(r"^(Patient_[0-9]+)_([0-9]+)$")


def load_spacing(path: Path) -> dict[str, tuple[float, float, float]]:
    with path.open("rb") as file:
        values = pickle.load(file)
    return {patient: tuple(float(item) for item in spacing)
            for patient, spacing in values.items()}


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    classes = datasets_params[args.dataset]["K"]
    validation = SliceDataset(
        "val", args.dataset_root, img_transform=img_transform,
        gt_transform=lambda image: gt_transform(classes, image),
        experiment="B0", context_slices=args.context_slices,
    )
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    model = TwoPointFiveD(
        1, classes, kernels=args.kernels, factor=args.factor,
        context_slices=args.context_slices, backbone=args.backbone,
    ).to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()

    predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    targets: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    with torch.no_grad():
        for data in loader:
            probabilities = F.softmax(model(data["images"].to(device)), dim=1)
            predicted = probabilities.argmax(dim=1).cpu().numpy()
            target = data["gts"].argmax(dim=1).cpu().numpy()
            for index, stem in enumerate(data["stems"]):
                match = PATIENT_SLICE.match(stem)
                if match is None:
                    raise ValueError(f"Unexpected slice stem: {stem}")
                patient, z = match.group(1), int(match.group(2))
                predictions[patient][z] = predicted[index]
                targets[patient][z] = target[index]

    spacing = load_spacing(args.spacing)
    rows: list[dict[str, object]] = []
    for patient in sorted(predictions):
        if set(predictions[patient]) != set(targets[patient]):
            raise ValueError(f"Slice mismatch for {patient}")
        indices = sorted(predictions[patient])
        prediction = np.stack([predictions[patient][z] for z in indices])
        target = np.stack([targets[patient][z] for z in indices])
        if patient not in spacing:
            raise KeyError(f"No spacing found for {patient}")
        # PNG volumes are stacked as (z, y, x), while spacing is stored as (x, y, z).
        metric_spacing = tuple(reversed(spacing[patient]))
        row: dict[str, object] = {"patient": patient, "slices": len(indices)}
        rve_values = []
        npd_values = []
        for class_id, class_name in enumerate(CLASS_NAMES[:classes]):
            pred_mask = prediction == class_id
            target_mask = target == class_id
            rve = float(relative_volume_error(
                pred_mask, target_mask, voxelspacing=metric_spacing,
            ).item())
            npd = float(normalized_surface_distance_single(
                pred_mask, target_mask, voxelspacing=metric_spacing,
                tau=args.tolerance,
            ))
            row[f"rve_{class_name.lower()}"] = rve
            row[f"npd_{class_name.lower()}"] = npd
            if class_id:
                rve_values.append(rve)
                npd_values.append(npd)
        row["rve_foreground_mean"] = float(np.mean(rve_values))
        row["npd_foreground_mean"] = float(np.mean(npd_values))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SEGTHOR_FIXED", choices=datasets_params)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spacing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=("unet", "vnet"), default="unet")
    parser.add_argument("--context-slices", type=int, default=3)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.context_slices < 1 or args.context_slices % 2 == 0:
        parser.error("--context-slices must be a positive odd number")
    rows = evaluate(args)
    foreground_rve = np.mean([row["rve_foreground_mean"] for row in rows])
    foreground_npd = np.mean([row["npd_foreground_mean"] for row in rows])
    print(f"Patients: {len(rows)}")
    print(f"Foreground RVE: {foreground_rve:.6f}")
    print(f"Foreground NPD: {foreground_npd:.6f}")
    print(f"Results: {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()