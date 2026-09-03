#!/usr/bin/env python3
"""Train a Palette-style conditional DDPM for IR -> RGB translation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pix2pix import PairedDataset


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, 1, 1)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 3, base: int = 128, emb_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 4), nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim * 4))
        self.in_conv = nn.Conv2d(in_channels, base, 3, 1, 1)
        self.down1 = nn.ModuleList([ResBlock(base, emb_dim * 4), ResBlock(base, emb_dim * 4)])
        self.down2 = nn.ModuleList([ResBlock(base, emb_dim * 4, base * 2), ResBlock(base * 2, emb_dim * 4)])
        self.down3 = nn.ModuleList([ResBlock(base * 2, emb_dim * 4, base * 2), ResBlock(base * 2, emb_dim * 4)])
        self.mid = nn.ModuleList([ResBlock(base * 2, emb_dim * 4), ResBlock(base * 2, emb_dim * 4)])
        self.up1 = nn.ModuleList([ResBlock(base * 4, emb_dim * 4, base), ResBlock(base, emb_dim * 4)])
        self.up2 = nn.ModuleList([ResBlock(base * 2, emb_dim * 4, base), ResBlock(base, emb_dim * 4)])
        self.out_norm = nn.GroupNorm(32, base)
        self.out_conv = nn.Conv2d(base, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, 128))
        h0 = self.in_conv(x)
        h = h0
        for block in self.down1:
            h = block(h, emb)
        skip1 = h
        h = F.avg_pool2d(h, 2)
        for block in self.down2:
            h = block(h, emb)
        skip2 = h
        h = F.avg_pool2d(h, 2)
        for block in self.down3:
            h = block(h, emb)
        for block in self.mid:
            h = block(h, emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, skip2], 1)
        for block in self.up1:
            h = block(h, emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.cat([h, skip1], 1)
        for block in self.up2:
            h = block(h, emb)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "palette")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--checkpointing-steps", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    betas = torch.linspace(1e-4, 0.02, args.num_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0).to(device)

    loader = DataLoader(
        PairedDataset(args.data / "train.csv", training=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(PairedDataset(args.data / "test.csv", training=False), batch_size=8, num_workers=args.workers)
    fixed_ir, _ = next(iter(test_loader))
    fixed_ir = fixed_ir[:8].to(device)

    step = 0
    while step < args.steps:
        for ir, rgb in loader:
            if step >= args.steps:
                break
            ir = ir.to(device)
            rgb = rgb.to(device)
            bsz = rgb.shape[0]
            t = torch.randint(0, args.num_timesteps, (bsz,), device=device)
            noise = torch.randn_like(rgb)
            a = alpha_cumprod[t][:, None, None, None]
            noisy = a.sqrt() * rgb + (1 - a).sqrt() * noise
            pred = model(torch.cat([noisy, ir], 1), t)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{args.steps} loss={loss.item():.6f}", flush=True)
            if step % args.checkpointing_steps == 0:
                torch.save({"model": model.state_dict(), "step": step}, args.output / f"checkpoint-{step}.pt")

    torch.save({"model": model.state_dict(), "step": args.steps}, args.output / "final.pt")
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
