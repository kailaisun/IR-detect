#!/usr/bin/env python3
"""Sample and evaluate a Palette-style conditional DDPM.

Supports ``--target thermal`` (1-channel z-scored field) and ``--target pseudo``
(3-channel pseudo-color). Uses deterministic DDIM sampling with a configurable
number of steps to keep evaluation tractable.
"""

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
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import RGB2TInferenceDataset, read_thermal
from train_palette import PaletteUNet


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    rgb = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    rows = [item[2] for item in batch]
    return rgb, target, rows


def build_schedule(num_timesteps: int = 1000) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


@torch.inference_mode()
def ddim_sample(
    model: PaletteUNet,
    cond: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    sampling_steps: int,
    out_channels: int,
) -> torch.Tensor:
    b = cond.shape[0]
    h, w = cond.shape[2], cond.shape[3]
    device = cond.device
    x = torch.randn(b, out_channels, h, w, device=device)

    num_timesteps = len(alpha_cumprod)
    times = torch.linspace(num_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device)

    for i in range(sampling_steps):
        t = times[i]
        t_prev = times[i + 1] if i + 1 < sampling_steps else -1
        t_batch = torch.full((b,), t.item(), device=device)
        eps = model(torch.cat([x, cond], dim=1), t_batch)

        a_t = alpha_cumprod[t]
        x0 = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-6)

        if t_prev < 0:
            x = x0
        else:
            a_prev = alpha_cumprod[t_prev]
            x = a_prev.sqrt() * x0 + (1.0 - a_prev).sqrt() * eps
    return x


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
    native = F.interpolate(pred, size=(62, 80), mode="area")[:, 0].cpu().numpy()
    return native * pixel_std + pixel_mean


def upscale_thermal(img: np.ndarray, size: tuple[int, int] = (256, 192)) -> np.ndarray:
    """Upscale a (H, W, 3) colormap image to match the RGB canvas (W, H)."""
    return np.asarray(Image.fromarray(img).resize(size, Image.NEAREST))


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    psnr = float(peak_signal_noise_ratio(b, a, data_range=255))
    ssim = float(structural_similarity(b, a, data_range=255, channel_axis=2))
    return psnr, ssim


def lpips_between(images_a: np.ndarray, images_b: np.ndarray, lpips_fn) -> float:
    def to_lpips(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
        return (t - 0.5) / 0.5

    a = to_lpips(images_a).to(next(lpips_fn.parameters()).device)
    b = to_lpips(images_b).to(next(lpips_fn.parameters()).device)
    with torch.inference_mode():
        values = lpips_fn(a, b, normalize=True)
    return float(values.mean().item())


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    stats = np.load(args.data / "thermal_stats.npz")
    pixel_mean = stats["pixel_mean"].astype(np.float32)
    pixel_std = stats["pixel_std"].astype(np.float32)

    out_channels = 1 if args.target == "thermal" else 3
    model = PaletteUNet(cond_channels=3, out_channels=out_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    alpha_cumprod = build_schedule().to(device)
    dataset = RGB2TInferenceDataset(args.data / "test.csv", target=args.target, training=False)
    if args.limit is not None:
        import torch.utils.data as tud

        dataset = tud.Subset(dataset, range(args.limit))
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

    is_field = args.target == "thermal"

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
    scenes: dict[str, dict] = {}
    sample_buf: list = []

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

    for rgb, target, rows in tqdm(loader, desc=f"sample {args.target}"):
        rgb = rgb.to(device, non_blocking=True)
        pred = ddim_sample(model, rgb, alpha_cumprod, args.sampling_steps, out_channels)

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

        if args.samples is not None and len(sample_buf) < 8:
            for i in range(b):
                if len(sample_buf) >= 8:
                    break
                if is_field:
                    sample_buf.append(
                        (rgb[i].cpu(), upscale_thermal(pred_rgb[i]), upscale_thermal(gt_rgb[i]))
                    )
                else:
                    sample_buf.append((rgb[i].cpu(), pred_u8[i], gt_u8[i]))

    metrics: dict = {}
    if is_field:
        metrics = {
            "temp_mae_c": g_sum_abs / g_px,
            "temp_rmse_c": float(np.sqrt(g_sum_sq / g_px)),
            "temp_r2": float(1.0 - g_sum_sq / (g_sum_gt_sq - g_sum_gt**2 / g_px)),
            "psnr": g_psnr / g_count,
            "ssim": g_ssim / g_count,
        }
    else:
        metrics = {"psnr": g_psnr / g_count, "ssim": g_ssim / g_count}

    if lpips_fn is not None and g_lpips_batches:
        metrics["lpips"] = g_lpips / g_lpips_batches

    per_scene: dict = {}
    for sid in sorted(scenes):
        sb = scenes[sid]
        entry = {"images": sb["images"], "psnr": sb["psnr"] / sb["images"], "ssim": sb["ssim"] / sb["images"]}
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
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_scene"}, indent=2))

    if args.samples is not None and sample_buf:
        Path(args.samples).parent.mkdir(parents=True, exist_ok=True)
        rgb_vis = [(s[0] + 1) / 2 for s in sample_buf]
        pred_vis = [torch.from_numpy(s[1]).permute(2, 0, 1).float() / 255.0 for s in sample_buf]
        gt_vis = [torch.from_numpy(s[2]).permute(2, 0, 1).float() / 255.0 for s in sample_buf]
        grid = torch.cat([torch.stack(rgb_vis), torch.stack(pred_vis), torch.stack(gt_vis)], dim=0)
        save_image(grid, args.samples, nrow=len(sample_buf))
        print("saved samples", args.samples)


if __name__ == "__main__":
    main()
