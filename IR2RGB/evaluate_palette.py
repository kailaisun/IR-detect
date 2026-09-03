#!/usr/bin/env python3
"""Evaluate the Palette DDPM with DDIM sampling on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import lpips
from train_palette import UNet


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "palette" / "final.pt")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "palette" / "eval")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def build_schedule(num_timesteps: int = 1000) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    return torch.cumprod(1.0 - betas, dim=0)


@torch.inference_mode()
def ddim_sample(model: UNet, cond: torch.Tensor, alpha_cumprod: torch.Tensor, steps: int) -> torch.Tensor:
    device = cond.device
    bsz = cond.shape[0]
    times = torch.linspace(len(alpha_cumprod) - 1, 0, steps + 1).long().to(device)
    x = torch.randn(bsz, 3, cond.shape[2], cond.shape[3], device=device)
    for i in range(steps):
        t = times[i]
        t_next = times[i + 1]
        t_batch = t.repeat(bsz)
        pred = model(torch.cat([x, cond], 1), t_batch)
        a = alpha_cumprod[t]
        x0 = (x - (1 - a).sqrt() * pred) / a.sqrt()
        if t_next >= 0:
            a_next = alpha_cumprod[t_next]
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * pred
        else:
            x = x0
    return x


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    alpha_cumprod = build_schedule().to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    with (args.test_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]

    psnr_sum = ssim_sum = lpips_sum = 0.0
    for i, row in enumerate(rows):
        ir_img = Image.open(row["ir_path"]).convert("RGB").resize((256, 192), Image.BILINEAR)
        rgb_img = Image.open(row["rgb_path"]).convert("RGB").resize((256, 192), Image.BILINEAR)
        ir = np.asarray(ir_img).astype(np.float32) / 255.0
        rgb = np.asarray(rgb_img).astype(np.float32) / 255.0
        ir_t = torch.from_numpy(ir).permute(2, 0, 1).unsqueeze(0).to(device)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        cond = (ir_t - 0.5) / 0.5
        gen = ddim_sample(model, cond, alpha_cumprod, args.ddim_steps)  # [-1,1]
        gen_01 = (gen + 1) / 2
        mse = F.mse_loss(gen_01, rgb_t)
        psnr_sum += float(10 * np.log10(1 / mse.item()))
        gen_np = (gen_01.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rgb_np = (rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ssim_sum += float(ssim(gen_np, rgb_np, channel_axis=2, data_range=255))
        lpips_sum += float(lpips_fn(gen, rgb_t * 2 - 1))
        if (i + 1) % 100 == 0:
            print(f"evaluated {i + 1}/{len(rows)}", flush=True)

    n = len(rows)
    metrics = {
        "images": n,
        "ddim_steps": args.ddim_steps,
        "psnr": round(psnr_sum / n, 4),
        "ssim": round(ssim_sum / n, 4),
        "lpips": round(lpips_sum / n, 4),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
