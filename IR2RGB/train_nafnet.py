#!/usr/bin/env python3
"""Train a NAFNet-style restoration network for IR -> RGB translation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pix2pix import PairedDataset


class LayerNorm2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[1:], None, None, 1e-6)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight(x)


class NAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels * 2)
        self.gate = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d()
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels * 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv3(self.sca(self.gate(self.conv2(self.conv1(self.norm1(x))))))
        x = residual + x * self.beta
        y = self.conv5(F.gelu(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, width: int = 64, blocks: int = 16) -> None:
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.intro(x)
        out = self.ending(self.body(residual))
        return torch.tanh(out + x)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "nafnet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--sample-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "samples").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = NAFNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(
        PairedDataset(args.data / "train.csv", training=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(PairedDataset(args.data / "test.csv", training=False), batch_size=8, num_workers=args.workers)
    fixed_ir, _ = next(iter(test_loader))
    fixed_ir = fixed_ir[:8].to(device)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for ir, rgb in loader:
            ir, rgb = ir.to(device), rgb.to(device)
            out = model(ir)
            loss = F.l1_loss(out, rgb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * ir.shape[0]
            n += ir.shape[0]
        scheduler.step()
        avg = total / n
        print(f"epoch {epoch}/{args.epochs} l1={avg:.6f}", flush=True)
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.inference_mode():
                sample = model(fixed_ir)
            save_image((torch.cat([fixed_ir, sample]) + 1) / 2, args.output / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
