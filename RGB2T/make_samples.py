#!/usr/bin/env python3
"""Generate consistent RGB -> thermal visualization grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from datasets import RGB2TInferenceDataset, RGB2TRelDataset, read_thermal
from unet import UNet


def colormap_celsius(field: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    vmin, vmax = 15.0, 35.0
    norm = (np.clip(field, vmin, vmax) - vmin) / (vmax - vmin)
    return (cm.inferno(norm)[..., :3] * 255).round().astype(np.uint8)


def rgb_tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def upscale_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((256, 192), Image.NEAREST))


def thermal_tensor_to_celsius(x: torch.Tensor, pixel_mean: np.ndarray, pixel_std: np.ndarray) -> np.ndarray:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = F.interpolate(x, size=(62, 80), mode="area")[0, 0].cpu().numpy()
    return x * pixel_std + pixel_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo", "relative"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    stats = np.load(Path(__file__).resolve().parent / "data" / "thermal_stats.npz")
    pixel_mean = stats["pixel_mean"].astype(np.float32)
    pixel_std = stats["pixel_std"].astype(np.float32)

    if args.target == "relative":
        ds = RGB2TRelDataset(Path(__file__).resolve().parent / "data" / "test.csv", training=False)
        model = UNet(in_channels=4, out_channels=1, base=64, activation="none").to(device)
        state_key = "generator" if "generator" in torch.load(args.checkpoint, map_location="cpu") else "model"
        rows = ds.rows
    else:
        ds = RGB2TInferenceDataset(Path(__file__).resolve().parent / "data" / "test.csv", target=args.target, training=False)
        out_channels = 1 if args.target == "thermal" else 3
        activation = "none" if args.target == "thermal" else "tanh"
        model = UNet(in_channels=3, out_channels=out_channels, base=64, activation=activation).to(device)
        state_key = "generator" if "generator" in torch.load(args.checkpoint, map_location="cpu") else "model"
        rows = ds.rows

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt[state_key])
    model.eval()

    inputs = []
    for idx in range(args.n):
        if args.target == "relative":
            cond, _ = ds[idx]
            cond = cond.to(device)
            rgb = cond[:3]
            with torch.inference_mode():
                pred = model(cond.unsqueeze(0))[0]
            pred_c = thermal_tensor_to_celsius(pred, pixel_mean, pixel_std)
            gt_c = read_thermal(Path(rows[idx]["thermal_path"]))
            rgb_img = rgb_tensor_to_numpy(rgb.unsqueeze(0))[0]
            pred_img = upscale_image(colormap_celsius(pred_c))
            gt_img = upscale_image(colormap_celsius(gt_c))
        elif args.target == "thermal":
            rgb, _, _ = ds[idx]
            rgb = rgb.to(device)
            with torch.inference_mode():
                pred = model(rgb.unsqueeze(0))[0]
            pred_c = thermal_tensor_to_celsius(pred, pixel_mean, pixel_std)
            gt_c = read_thermal(Path(rows[idx]["thermal_path"]))
            rgb_img = rgb_tensor_to_numpy(rgb.unsqueeze(0))[0]
            pred_img = upscale_image(colormap_celsius(pred_c))
            gt_img = upscale_image(colormap_celsius(gt_c))
        else:
            rgb, _, _ = ds[idx]
            rgb = rgb.to(device)
            with torch.inference_mode():
                pred = model(rgb.unsqueeze(0))[0]
            ir = Image.open(rows[idx]["ir_path"]).convert("RGB").resize((256, 192), Image.BILINEAR)
            gt_img = np.asarray(ir)
            rgb_img = rgb_tensor_to_numpy(rgb.unsqueeze(0))[0]
            pred_img = rgb_tensor_to_numpy(pred.unsqueeze(0))[0]

        inputs.append((rgb_img, pred_img, gt_img))

    grid_rgb = np.concatenate([x[0] for x in inputs], axis=1)
    grid_pred = np.concatenate([x[1] for x in inputs], axis=1)
    grid_gt = np.concatenate([x[2] for x in inputs], axis=1)
    grid = np.concatenate([grid_rgb, grid_pred, grid_gt], axis=0)
    Image.fromarray(grid).save(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
