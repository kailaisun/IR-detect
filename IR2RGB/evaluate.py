#!/usr/bin/env python3
"""Evaluate Pix2Pix IR->RGB with a full metric suite on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

import lpips
from pytorch_msssim import ms_ssim

from pix2pix import PairedDataset, UNetGenerator


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "runs" / "pix2pix" / "checkpoint_epoch_100.pt")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "pix2pix" / "eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def polynomial_mmd(x: np.ndarray, y: np.ndarray, degree: int = 3, coef0: float = 1.0) -> float:
    gamma = 1.0 / x.shape[1]
    x = np.sqrt(gamma) * x
    y = np.sqrt(gamma) * y
    kxx = (coef0 + x @ x.T) ** degree
    kyy = (coef0 + y @ y.T) ** degree
    kxy = (coef0 + x @ y.T) ** degree
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def kernel_inception_distance(real: np.ndarray, fake: np.ndarray, n_subsets: int = 100, subset_size: int = 1000) -> float:
    rng = np.random.RandomState(42)
    n = min(len(real), len(fake))
    values = []
    for _ in range(n_subsets):
        ri = rng.choice(n, size=subset_size, replace=True)
        fi = rng.choice(n, size=subset_size, replace=True)
        values.append(polynomial_mmd(real[ri], fake[fi]))
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    generator = UNetGenerator(in_channels=3, out_channels=3, num_downs=6).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    inception.eval()
    feat_real: list[torch.Tensor] = []
    feat_fake: list[torch.Tensor] = []
    current = [feat_real]

    def hook(module, inputs, output):
        current[0].append(output.flatten(1))

    handle = inception.avgpool.register_forward_hook(hook)

    rows = list(csv.DictReader((args.data / "test.csv").open()))
    loader = DataLoader(
        PairedDataset(args.data / "test.csv", training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    inception_resize = transforms.Resize((299, 299), antialias=True)
    inception_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    sums = {
        "psnr": 0.0,
        "ssim": 0.0,
        "ms_ssim": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "lpips": 0.0,
        "deltaE": 0.0,
    }
    per_scene: dict[str, dict[str, float]] = defaultdict(lambda: {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0})
    count = 0
    sample_ir, sample_fake, sample_real = None, None, None

    with torch.inference_mode():
        for batch_idx, (ir, rgb) in enumerate(loader):
            ir = ir.to(device)
            rgb = rgb.to(device)
            fake = generator(ir)

            fake_01 = (fake + 1) / 2
            rgb_01 = (rgb + 1) / 2
            b = ir.shape[0]

            mse = F.mse_loss(fake_01, rgb_01)
            sums["psnr"] += float(10.0 * np.log10(1.0 / mse.item())) * b
            sums["mse"] += float(mse.item()) * b
            sums["mae"] += float(F.l1_loss(fake_01, rgb_01).item()) * b
            sums["ms_ssim"] += float(ms_ssim(fake_01, rgb_01, data_range=1.0, size_average=True)) * b
            sums["lpips"] += float(lpips_fn(fake, rgb).sum())

            fake_np = (fake_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            rgb_np = (rgb_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            for f, r in zip(fake_np, rgb_np, strict=True):
                sums["ssim"] += float(ssim(f, r, channel_axis=2, data_range=255))
                sums["deltaE"] += float(deltaE_ciede2000(rgb2lab(r), rgb2lab(f)).mean())

            # Inception features for FID / KID.
            real_in = inception_norm(inception_resize(rgb_01))
            fake_in = inception_norm(inception_resize(fake_01))
            current[0] = feat_real
            _ = inception(real_in)
            current[0] = feat_fake
            _ = inception(fake_in)

            # Per-scene accumulation.
            scene_batch = [rows[batch_idx * args.batch_size + i]["scene_id"] for i in range(b)]
            for i, scene in enumerate(scene_batch):
                per_scene[scene]["psnr"] += float(10.0 * np.log10(1.0 / max(F.mse_loss(fake_01[i], rgb_01[i]).item(), 1e-12)))
                per_scene[scene]["ssim"] += float(ssim(fake_np[i], rgb_np[i], channel_axis=2, data_range=255))
                per_scene[scene]["count"] += 1

            count += b
            if sample_fake is None:
                sample_ir, sample_fake, sample_real = ir[:8], fake[:8], rgb[:8]

    handle.remove()

    real_feat = torch.cat(feat_real, 0).cpu().numpy()
    fake_feat = torch.cat(feat_fake, 0).cpu().numpy()
    mu_real, mu_fake = real_feat.mean(0), fake_feat.mean(0)
    sigma_real = np.cov(real_feat.T)
    sigma_fake = np.cov(fake_feat.T)
    diff = mu_real - mu_fake
    eigvals = np.linalg.eigvals(sigma_real @ sigma_fake)
    fid = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * np.sqrt(np.clip(eigvals.real, 0, None)).sum())
    kid = kernel_inception_distance(real_feat, fake_feat)

    metrics = {
        "images": count,
        "psnr": round(sums["psnr"] / count, 4),
        "ssim": round(sums["ssim"] / count, 4),
        "ms_ssim": round(sums["ms_ssim"] / count, 4),
        "mae": round(sums["mae"] / count, 4),
        "rmse": round((sums["mse"] / count) ** 0.5, 4),
        "lpips": round(sums["lpips"] / count, 4),
        "deltaE_ciede2000": round(sums["deltaE"] / count, 4),
        "fid": round(fid, 4),
        "kid": round(kid, 6),
        "per_scene": {
            scene: {
                "images": int(v["count"]),
                "psnr": round(v["psnr"] / v["count"], 4),
                "ssim": round(v["ssim"] / v["count"], 4),
            }
            for scene, v in sorted(per_scene.items())
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    grid = torch.cat([sample_ir, sample_fake, sample_real], 0)
    grid = (grid + 1) / 2
    save_image(grid, args.output / "samples.png", nrow=8)
    print("samples saved to", args.output / "samples.png")


if __name__ == "__main__":
    main()
