#!/usr/bin/env python3
"""Evaluate an RGB -> thermal-field or pseudo-color model on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import IMAGE_H, IMAGE_W, RGB2TInferenceDataset, read_thermal
from unet import UNet


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    rgb = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    rows = [item[2] for item in batch]
    return rgb, target, rows


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    x = x.clamp(-1, 1)
    x = (x + 1) / 2
    x = (x * 255).round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()


def field_to_colormap(field: np.ndarray) -> np.ndarray:
    """Map a Celsius field to an RGB uint8 image with a fixed inferno colormap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    vmin, vmax = 15.0, 35.0
    norm = (np.clip(field, vmin, vmax) - vmin) / (vmax - vmin)
    rgb = cm.inferno(norm)[..., :3]
    return (rgb * 255).round().astype(np.uint8)


def lpips_between(images_a: np.ndarray, images_b: np.ndarray, lpips_fn) -> float:
    def to_lpips(img: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
        return (tensor - 0.5) / 0.5

    a = to_lpips(images_a).to(next(lpips_fn.parameters()).device)
    b = to_lpips(images_b).to(next(lpips_fn.parameters()).device)
    with torch.inference_mode():
        values = lpips_fn(a, b, normalize=True)
    return float(values.mean().item())


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    stats = np.load(args.data / "thermal_stats.npz")
    pixel_mean = stats["pixel_mean"].astype(np.float32)
    pixel_std = stats["pixel_std"].astype(np.float32)

    out_channels = 1 if args.target == "thermal" else 3
    activation = "none" if args.target == "thermal" else "tanh"
    model = UNet(in_channels=3, out_channels=out_channels, base=64, activation=activation).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("generator"))
    if state is None:
        raise RuntimeError(f"checkpoint {args.checkpoint} has no model/generator key")
    model.load_state_dict(state)
    model.eval()

    dataset = RGB2TInferenceDataset(args.data / "test.csv", target=args.target, training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate,
    )

    lpips_fn = None
    if args.target == "pseudo" or True:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)

    scene = {}
    all_psnr, all_ssim, all_lpips = [], [], []
    all_mae_c, all_rmse_c, all_r2 = [], [], []
    sample_rows = []

    for rgb, target, rows in tqdm(loader, desc="eval"):
        rgb = rgb.to(device, non_blocking=True)
        with torch.inference_mode():
            pred = model(rgb)

        if args.target == "thermal":
            pred_u8 = tensor_to_uint8(pred.repeat(1, 3, 1, 1))
            target_u8 = tensor_to_uint8(target.repeat(1, 3, 1, 1))
            for i in range(pred.shape[0]):
                p = pred[i : i + 1]
                p_native = F.interpolate(p, size=(62, 80), mode="area")[0, 0].cpu().numpy()
                p_c = p_native * pixel_std + pixel_mean
                gt_c = read_thermal(Path(rows[i]["thermal_path"]))
                diff = p_c - gt_c
                mae = float(np.abs(diff).mean())
                rmse = float(np.sqrt((diff**2).mean()))
                ss_res = float((diff**2).sum())
                ss_tot = float(((gt_c - gt_c.mean()) ** 2).sum())
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                all_mae_c.append(mae)
                all_rmse_c.append(rmse)
                all_r2.append(r2)
                sid = rows[i]["scene_id"]
                bucket = scene.setdefault(sid, {"images": 0, "mae_c": 0.0, "rmse_c": 0.0})
                bucket["images"] += 1
                bucket["mae_c"] += mae
                bucket["rmse_c"] += rmse
                if len(sample_rows) < 8:
                    sample_rows.append((rgb[i : i + 1], pred[i : i + 1], target[i : i + 1], rows[i]))
        else:
            pred_u8 = tensor_to_uint8(pred)
            target_u8 = tensor_to_uint8(target)
            if len(sample_rows) < 8:
                sample_rows.append((rgb[0:1], pred[0:1], target[0:1], rows[0]))

        # Per-image image metrics on the rendered/normalized domain.
        for i in range(pred_u8.shape[0]):
            p = pred_u8[i]
            g = target_u8[i]
            if args.target == "pseudo":
                psnr = float(peak_signal_noise_ratio(g, p, data_range=255))
                ssim = float(structural_similarity(g, p, data_range=255, channel_axis=2))
            else:
                # For thermal, compare the upsampled Celsius field through its colormap.
                p_native = F.interpolate(pred[i : i + 1], size=(62, 80), mode="area")[0, 0].cpu().numpy()
                p_c = p_native * pixel_std + pixel_mean
                gt_c = read_thermal(Path(rows[i]["thermal_path"]))
                p_cmap = field_to_colormap(p_c)
                g_cmap = field_to_colormap(gt_c)
                psnr = float(peak_signal_noise_ratio(g_cmap, p_cmap, data_range=255))
                ssim = float(structural_similarity(g_cmap, p_cmap, data_range=255, channel_axis=2))
            all_psnr.append(psnr)
            all_ssim.append(ssim)

        if lpips_fn is not None:
            if args.target == "thermal":
                # colormap rendering for LPIPS on a subset to keep it fast.
                subset = []
                for i in range(min(256, pred.shape[0])):
                    p_native = F.interpolate(pred[i : i + 1], size=(62, 80), mode="area")[0, 0].cpu().numpy()
                    p_c = p_native * pixel_std + pixel_mean
                    gt_c = read_thermal(Path(rows[i]["thermal_path"]))
                    subset.append((field_to_colormap(p_c), field_to_colormap(gt_c)))
                if subset:
                    a = np.stack([x[0] for x in subset])
                    b = np.stack([x[1] for x in subset])
                    all_lpips.append(lpips_between(a, b, lpips_fn))
            else:
                all_lpips.append(lpips_between(pred_u8, target_u8, lpips_fn))

    metrics: dict[str, float] = {}
    if args.target == "thermal":
        metrics = {
            "temp_mae_c": float(np.mean(all_mae_c)),
            "temp_rmse_c": float(np.mean(all_rmse_c)),
            "temp_r2": float(np.mean(all_r2)),
        }
    metrics.update(
        {
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
            "lpips": float(np.mean(all_lpips)) if all_lpips else None,
        }
    )
    metrics["per_scene"] = {
        sid: {
            "images": v["images"],
            **({"mae_c": v["mae_c"] / v["images"], "rmse_c": v["rmse_c"] / v["images"]} if args.target == "thermal" else {}),
        }
        for sid, v in sorted(scene.items())
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if sample_rows:
        save_n = min(8, len(sample_rows))
        rgb_vis = torch.stack([s[0] for s in sample_rows[:save_n]]).to(device)
        pred_vis = torch.stack([s[1] for s in sample_rows[:save_n]]).to(device)
        gt_vis = torch.stack([s[2] for s in sample_rows[:save_n]]).to(device)
        if rgb_vis.ndim == 5:
            rgb_vis = rgb_vis.squeeze(2)
        if pred_vis.ndim == 5:
            pred_vis = pred_vis.squeeze(2)
        if gt_vis.ndim == 5:
            gt_vis = gt_vis.squeeze(2)
        rgb_vis = (rgb_vis + 1) / 2
        if args.target == "thermal":
            pred_vis = (pred_vis - pred_vis.min()) / (pred_vis.max() - pred_vis.min() + 1e-8)
            gt_vis = (gt_vis - gt_vis.min()) / (gt_vis.max() - gt_vis.min() + 1e-8)
        else:
            pred_vis = (pred_vis + 1) / 2
            gt_vis = (gt_vis + 1) / 2
        pred_vis = pred_vis.repeat(1, 3, 1, 1) if pred_vis.shape[1] == 1 else pred_vis
        gt_vis = gt_vis.repeat(1, 3, 1, 1) if gt_vis.shape[1] == 1 else gt_vis
        grid = torch.cat([rgb_vis, pred_vis, gt_vis], 0)
        save_image(grid, out / "samples.png", nrow=save_n)


if __name__ == "__main__":
    main()
