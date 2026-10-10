#!/usr/bin/env python3
"""Run prediction-only inference with any supported segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ENet import ENet
from ResUNetPP import ResUNetPP
from ShallowNet import shallowCNN
from TwoPointFiveD import TwoPointFiveD
from UNet import UNet
from UNet3D import UNet3D
from dataset import SliceDataset, VolumeDataset
from main import datasets_params, img_transform
from utils import probs2class


MODEL_CLASSES = {
    "shallowcnn": shallowCNN,
    "enet": ENet,
    "unet": UNet,
    "unet25d": TwoPointFiveD,
    "unet3d": UNet3D,
    "resunetpp": ResUNetPP,
}


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict", checkpoint.get("model_state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    model.load_state_dict(state_dict)


def save_slice_predictions(predictions: torch.Tensor, stems: list[str],
                           destination: Path) -> None:
    for prediction, stem in zip(predictions.cpu(), stems):
        Image.fromarray(prediction.numpy().astype("uint8")).save(
            destination / f"{stem}.png"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="A bestweights.pt or compatible checkpoint.")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Sliced dataset root containing test/img.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory for class-index PNG predictions.")
    parser.add_argument("--dataset", choices=tuple(datasets_params),
                        default="SEGTHOR_FULL")
    parser.add_argument("--context-slices", type=int, default=3)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.model == "unet25d" and (
        args.context_slices < 1 or args.context_slices % 2 == 0
    ):
        parser.error("--context-slices must be a positive odd number")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    classes = datasets_params[args.dataset]["K"]
    is_3d = args.model == "unet3d"
    if is_3d:
        dataset = VolumeDataset("test", args.dataset_root,
                                img_transform=img_transform, debug=args.debug)
        input_channels = 1
    else:
        context_slices = args.context_slices if args.model == "unet25d" else 1
        dataset = SliceDataset(
            "test", args.dataset_root, img_transform=img_transform,
            debug=args.debug, context_slices=context_slices,
        )
        input_channels = 1

    loader = DataLoader(
        dataset, batch_size=1 if is_3d else args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    model_kwargs = {
        "kernels": args.kernels,
        "factor": args.factor,
    }
    if args.model == "unet25d":
        model_kwargs["context_slices"] = args.context_slices
    model = MODEL_CLASSES[args.model](
        input_channels, classes, **model_kwargs
    ).to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for data in loader:
            images = data["images"].to(device)
            predictions = probs2class(F.softmax(model(images), dim=1))
            if is_3d:
                for prediction, stem in zip(predictions.cpu(), data["stems"]):
                    save_slice_predictions(
                        prediction, [f"{stem}_{index:04d}" for index in range(prediction.shape[0])],
                        args.output_root,
                    )
            else:
                save_slice_predictions(predictions, list(data["stems"]),
                                       args.output_root)
    print(f"Saved {len(list(args.output_root.glob('*.png')))} predictions to {args.output_root}")


if __name__ == "__main__":
    main()
