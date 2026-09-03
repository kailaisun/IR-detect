#!/usr/bin/env python3
"""Minimal SD1.5 ControlNet training loop for IR -> RGB translation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parent
RES = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "sd15")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "controlnet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--checkpointing-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ControlNetDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        with csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(RES),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(RES),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        ir = Image.open(row["ir_path"]).convert("RGB")
        return {
            "pixel_values": self.image_transform(rgb),
            "conditioning_pixel_values": self.cond_transform(ir),
        }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = torch.float32
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet").to(device, dtype=weight_dtype)
    controlnet = ControlNetModel.from_unet(unet).to(device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    controlnet.train()

    empty_ids = tokenizer(
        [""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    with torch.inference_mode():
        empty_embeds = text_encoder(empty_ids)[0]  # (1, 77, 768)

    loader = DataLoader(
        ControlNetDataset(args.train_csv),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in loader:
            if global_step >= args.max_train_steps:
                break
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            cond_values = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
            bsz = pixel_values.shape[0]
            embeds = empty_embeds.repeat(bsz, 1, 1)

            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            down_res, mid_res = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeds,
                controlnet_cond=cond_values,
                return_dict=False,
            )
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeds,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred, noise, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.6f}", flush=True)
            if global_step % args.checkpointing_steps == 0:
                controlnet.save_pretrained(args.output / f"checkpoint-{global_step}")
                print(f"saved checkpoint-{global_step}", flush=True)

    controlnet.save_pretrained(args.output / "final")
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
