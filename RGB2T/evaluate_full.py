#!/usr/bin/env python3
"""Complete evaluation for RGB -> thermal-field / pseudo-color / relative-thermal.

Computes global and per-scene metrics and writes one machine-readable JSON.

Targets
-------
* ``thermal``  : 3-channel RGB -> 1-channel z-scored temperature field.
* ``pseudo``   : 3-channel RGB -> 3-channel pseudo-color thermal image.
* ``relative`` : 4-channel [RGB, mean-field baseline] -> 1-channel z-scored
                 field (ThermalGAN-style).

Metrics
-------
* Thermal targets: temperature MAE/RMSE/R^2 (degrees C), plus PSNR/SSIM/LPIPS
  computed on an inferno colormap rendering.
* Pseudo target: PSNR/SSIM/LPIPS on RGB.
* Every target also gets a per-scene (room04-room12) breakdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    RGB2TInferenceDataset,
    RGB2TRelInferenceDataset,
    read_thermal,
)
from unet import UNet


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo", "relative"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    cond = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    rows = [item[2] for item in batch]
    return cond, target, rows


def colormap_celsius(field: np.ndarray, vmin: float = 15.0, vmax: float = 35.0) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    norm = (np.clip(field, vmin, vmax) - vmin) / (vmax - vmin)
    rgb = cm.inferno(norm)[..., :3]
    return (rgb * 255).round().astype(np.uint8)


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().clamp(-1, 1)
    x = (x + 1) / 2
    x = (x * 255).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()


def field_to_celsius(pred: torch.Tensor, pixel_mean: np.ndarray, pixel_std: np.ndarray) -> np.ndarray:
    """Downsample a predicted z-scored field tensor back to native (62,80) Celsius."""
    native = F.interpolate(pred, size=(62, 80), mode="area")[:, 0].cpu().numpy()
    return native * pixel_std + pixel_mean


def lpips_between(images_a: np.ndarray, images_b: np.ndarray, lpips_fn) -> float:
    def to_lpips(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
        return (t - 0.5) / 0.5

    a = to_lpips(images_a).to(next(lpips_fn.parameters()).device)
    b = to_lpips(images_b).to(next(lpips_fn.parameters()).device)
    with torch.inference_mode():
        values = lpips_fn(a, b, normalize=True)
    return float(values.mean().item())


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    psnr = float(peak_signal_noise_ratio(b, a, data_range=255))
    ssim = float(structural_similarity(b, a, data_range=255, channel_axis=2))
    return psnr, ssim


def build_model(target: str, device: torch.device, checkpoint: Path) -> UNet:
    if target == "relative":
        model = UNet(in_channels=4, out_channels=1, base=64, activation="none")
    elif target == "thermal":
        model = UNet(in_channels=3, out_channels=1, base=64, activation="none")
    else:
        model = UNet(in_channels=3, out_channels=3, base=64, activation="tanh")

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("generator"))
    if state is None:
        raise RuntimeError(f"checkpoint {checkpoint} has no model/generator key")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_dataset(target: str, data: Path):
    if target == "relative":
        return RGB2TRelInferenceDataset(data / "test.csv", training=False)
    return RGB2TInferenceDataset(data / "test.csv", target=target, training=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    stats = np.load(args.data / "thermal_stats.npz")
    pixel_mean = stats["pixel_mean"].astype(np.float32)
    pixel_std = stats["pixel_std"].astype(np.float32)

    model = build_model(args.target, device, args.checkpoint)
    dataset = build_dataset(args.target, args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )

    lpips_fn = None
    if not args.no_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)

    is_field = args.target in {"thermal", "relative"}

    # Global accumulators.
    g_count = 0
    g_px = 0
    g_sum_abs = 0.0
    g_sum_sq = 0.0
    g_sum_gt = 0.0
    g_sum_gt_sq = 0.0
    g_psnr = 0.0
    g_ssim = 0.0
    g_lpips = 0.0
    g_lpips_batches = 0

    # Per-scene accumulators.
    scenes: dict[str, dict] = {}

    def scene_bucket(sid: str) -> dict:
        return scenes.setdefault(
            sid,
            {
                "images": 0,
                "pixels": 0,
                "sum_abs": 0.0,
                "sum_sq": 0.0,
                "sum_gt": 0.0,
                "sum_gt_sq": 0.0,
                "psnr": 0.0,
                "ssim": 0.0,
            },
        )

    for cond, target, rows in tqdm(loader, desc=f"eval {args.target}"):
        cond = cond.to(device, non_blocking=True)
        with torch.inference_mode():
            pred = model(cond)

        b = pred.shape[0]
        if is_field:
            pred_c = field_to_celsius(pred, pixel_mean, pixel_std)
            pred_rgb = np.stack([colormap_celsius(f) for f in pred_c])
            gt_c = np.stack([read_thermal(Path(r["thermal_path"])) for r in rows])
            gt_rgb = np.stack([colormap_celsius(f) for f in gt_c])

            diff = pred_c - gt_c
            abs_diff = np.abs(diff)
            sq_diff = diff**2

            g_count += b
            g_px += diff.size
            g_sum_abs += float(abs_diff.sum())
            g_sum_sq += float(sq_diff.sum())
            g_sum_gt += float(gt_c.sum())
            g_sum_gt_sq += float((gt_c**2).sum())

            for i in range(b):
                sid = rows[i]["scene_id"]
                sb = scene_bucket(sid)
                sb["images"] += 1
                sb["pixels"] += diff[i].size
                sb["sum_abs"] += float(abs_diff[i].sum())
                sb["sum_sq"] += float(sq_diff[i].sum())
                sb["sum_gt"] += float(gt_c[i].sum())
                sb["sum_gt_sq"] += float((gt_c[i] ** 2).sum())
                p, s = psnr_ssim(pred_rgb[i], gt_rgb[i])
                sb["psnr"] += p
                sb["ssim"] += s

            # Per-image PSNR/SSIM for the global average.
            for i in range(b):
                p, s = psnr_ssim(pred_rgb[i], gt_rgb[i])
                g_psnr += p
                g_ssim += s
        else:
            pred_u8 = tensor_to_uint8(pred)
            gt_u8 = tensor_to_uint8(target)
            for i in range(b):
                sid = rows[i]["scene_id"]
                sb = scene_bucket(sid)
                sb["images"] += 1
                p, s = psnr_ssim(pred_u8[i], gt_u8[i])
                sb["psnr"] += p
                sb["ssim"] += s
                g_psnr += p
                g_ssim += s
                g_count += 1

        if lpips_fn is not None:
            if is_field:
                val = lpips_between(pred_rgb, gt_rgb, lpips_fn)
            else:
                val = lpips_between(pred_u8, gt_u8, lpips_fn)
            g_lpips += val
            g_lpips_batches += 1

    metrics: dict = {}
    if is_field:
        g_mae = g_sum_abs / g_px
        g_rmse = float(np.sqrt(g_sum_sq / g_px))
        g_ss_tot = g_sum_gt_sq - (g_sum_gt**2) / g_px
        g_r2 = 1.0 - g_sum_sq / g_ss_tot if g_ss_tot > 0 else 0.0
        metrics = {
            "temp_mae_c": g_mae,
            "temp_rmse_c": g_rmse,
            "temp_r2": float(g_r2),
            "psnr": g_psnr / g_count,
            "ssim": g_ssim / g_count,
        }
    else:
        metrics = {
            "psnr": g_psnr / g_count,
            "ssim": g_ssim / g_count,
        }

    if lpips_fn is not None and g_lpips_batches:
        metrics["lpips"] = g_lpips / g_lpips_batches

    per_scene: dict = {}
    for sid in sorted(scenes):
        sb = scenes[sid]
        entry = {
            "images": sb["images"],
            "psnr": sb["psnr"] / sb["images"],
            "ssim": sb["ssim"] / sb["images"],
        }
        if is_field:
            ss_tot = sb["sum_gt_sq"] - (sb["sum_gt"] ** 2) / sb["pixels"]
            entry["mae_c"] = sb["sum_abs"] / sb["pixels"]
            entry["rmse_c"] = float(np.sqrt(sb["sum_sq"] / sb["pixels"]))
            entry["r2"] = 1.0 - sb["sum_sq"] / ss_tot if ss_tot > 0 else 0.0
        per_scene[sid] = entry

    metrics["per_scene"] = per_scene
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
