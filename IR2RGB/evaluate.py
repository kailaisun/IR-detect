#!/usr/bin/env python3
"""Evaluate Pix2Pix IR->RGB on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

import lpips

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
    parser.add_argument("--fid-limit", type=int, default=0)
    return parser.parse_args()


def psnr_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b)
    return float(10.0 * np.log10(1.0 / mse.item()))


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

    test_set = PairedDataset(args.data / "test.csv", training=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    inception_resize = transforms.Resize((299, 299), antialias=True)
    inception_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    count = 0
    sample_ir, sample_fake, sample_real = None, None, None

    with torch.inference_mode():
        for ir, rgb in loader:
            ir = ir.to(device)
            rgb = rgb.to(device)
            fake = generator(ir)

            fake_01 = (fake + 1) / 2
            rgb_01 = (rgb + 1) / 2
            psnr_sum += psnr_tensor(fake_01, rgb_01) * ir.shape[0]

            fake_np = (fake_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            rgb_np = (rgb_01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            for f, r in zip(fake_np, rgb_np, strict=True):
                ssim_sum += float(ssim(f, r, channel_axis=2, data_range=255))

            lpips_sum += float(lpips_fn(fake, rgb).sum())
            count += ir.shape[0]

            # Inception features for FID.
            real_in = inception_norm(inception_resize(rgb_01))
            fake_in = inception_norm(inception_resize(fake_01))
            current[0] = feat_real
            _ = inception(real_in)
            current[0] = feat_fake
            _ = inception(fake_in)

            if sample_fake is None:
                sample_ir = ir[:8]
                sample_fake = fake[:8]
                sample_real = rgb[:8]

    handle.remove()

    real_feat = torch.cat(feat_real, 0)
    fake_feat = torch.cat(feat_fake, 0)
    mu_real = real_feat.mean(0)
    mu_fake = fake_feat.mean(0)
    sigma_real = torch.cov(real_feat.T)
    sigma_fake = torch.cov(fake_feat.T)
    diff = mu_real - mu_fake
    eigvals = torch.linalg.eigvals(sigma_real @ sigma_fake)
    fid = float(diff @ diff + sigma_real.trace() + sigma_fake.trace() - 2 * torch.sqrt(eigvals.real.clamp(min=0)).sum())

    metrics = {
        "images": count,
        "psnr": round(psnr_sum / count, 4),
        "ssim": round(ssim_sum / count, 4),
        "lpips": round(lpips_sum / count, 4),
        "fid": round(fid, 4),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    grid = torch.cat([sample_ir, sample_fake, sample_real], 0)
    grid = (grid + 1) / 2
    save_image(grid, args.output / "samples.png", nrow=8)
    print("samples saved to", args.output / "samples.png")


if __name__ == "__main__":
    main()
