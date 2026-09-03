#!/usr/bin/env python3
"""Minimal SDXL ControlNet training loop for IR -> RGB translation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from train_controlnet_local import ControlNetDataset


ROOT = Path(__file__).resolve().parent
RES = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "sdxl")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "controlnet_sdxl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-steps", type=int, default=5000)
    parser.add_argument("--checkpointing-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_dir, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model_dir, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.model_dir, subfolder="text_encoder", torch_dtype=torch.float32).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.model_dir, subfolder="text_encoder_2", torch_dtype=torch.float32
    ).to(device)
    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float32).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_dir, subfolder="unet", torch_dtype=torch.float32).to(device)
    controlnet = ControlNetModel.from_unet(unet).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")

    for model in (text_encoder, text_encoder_2, vae, unet):
        model.requires_grad_(False).eval()
    controlnet.train()

    ids1 = tokenizer([""], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    ids2 = tokenizer_2([""], max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out1 = text_encoder(ids1, output_hidden_states=True)
        out2 = text_encoder_2(ids2, output_hidden_states=True)
        prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)  # (1,77,2048)
        pooled = out2[0]  # (1,1280)
    time_ids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=device, dtype=pooled.dtype)

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
            pixel_values = batch["pixel_values"].to(device)
            cond_values = batch["conditioning_pixel_values"].to(device)
            bsz = pixel_values.shape[0]
            embeds = prompt_embeds.repeat(bsz, 1, 1)
            add_kwargs = {"text_embeds": pooled.repeat(bsz, 1), "time_ids": time_ids.repeat(bsz, 1)}

            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            down_res, mid_res = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeds,
                controlnet_cond=cond_values,
                added_cond_kwargs=add_kwargs,
                return_dict=False,
            )
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=embeds,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                added_cond_kwargs=add_kwargs,
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step % 25 == 0:
                print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.6f}", flush=True)
            if global_step % args.checkpointing_steps == 0:
                controlnet.save_pretrained(args.output / f"checkpoint-{global_step}")

    controlnet.save_pretrained(args.output / "final")
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
