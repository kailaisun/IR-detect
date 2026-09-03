#!/usr/bin/env python3
"""Evaluate a trained SD1.5 ControlNet (IR -> RGB) on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import lpips
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


ROOT = Path(__file__).resolve().parent
RES = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlnet", type=Path, default=ROOT / "runs" / "controlnet" / "checkpoint-10000")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "sd15")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "controlnet" / "eval")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def square(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = RES / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    left = (image.width - RES) // 2
    top = (image.height - RES) // 2
    return image.crop((left, top, left + RES, top + RES))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.model_dir, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(args.device)

    with (args.test_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]

    lpips_fn = lpips.LPIPS(net="alex").to(args.device)
    psnr_sum = ssim_sum = lpips_sum = 0.0
    sample_ir, sample_fake, sample_real = [], [], []

    for i, row in enumerate(rows):
        ir = Image.open(row["ir_path"]).convert("RGB")
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        ir = square(ir)
        rgb = square(rgb)
        gen = pipe(prompt="", image=ir, num_inference_steps=args.steps).images[0]

        ir_t = torch.from_numpy(np.asarray(ir).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device)
        gen_t = torch.from_numpy(np.asarray(gen).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device)
        rgb_t = torch.from_numpy(np.asarray(rgb).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(args.device)

        mse = torch.nn.functional.mse_loss(gen_t, rgb_t)
        psnr_sum += float(10 * np.log10(1 / mse.item()))
        ssim_sum += float(ssim(np.asarray(gen), np.asarray(rgb), channel_axis=2, data_range=255))
        lpips_sum += float(lpips_fn(gen_t * 2 - 1, rgb_t * 2 - 1))

        if i < 4:
            sample_ir.append(np.asarray(ir))
            sample_fake.append(np.asarray(gen))
            sample_real.append(np.asarray(rgb))
        if (i + 1) % 100 == 0:
            print(f"evaluated {i + 1}/{len(rows)}", flush=True)

    n = len(rows)
    metrics = {
        "images": n,
        "inference_steps": args.steps,
        "psnr": round(psnr_sum / n, 4),
        "ssim": round(ssim_sum / n, 4),
        "lpips": round(lpips_sum / n, 4),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    strips = []
    for ir, fake, real in zip(sample_ir, sample_fake, sample_real, strict=True):
        strips.append(np.concatenate([ir, fake, real], axis=1))
    montage = np.concatenate(strips, axis=0)
    Image.fromarray(montage).save(args.output / "samples.png")
    print("samples saved to", args.output / "samples.png")


if __name__ == "__main__":
    main()
