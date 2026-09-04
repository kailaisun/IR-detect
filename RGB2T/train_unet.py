#!/usr/bin/env python3
"""Train a deterministic U-Net for RGB -> thermal-field or pseudo-color."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import RGB2TDataset
from unet import UNet


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo"], default="thermal")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "unet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--loss", choices=["l1", "huber"], default="l1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def to_rgb_batch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    out_channels = 1 if args.target == "thermal" else 3
    activation = "none" if args.target == "thermal" else "tanh"
    model = UNet(in_channels=3, out_channels=out_channels, base=64, activation=activation).to(device)

    train_set = RGB2TDataset(args.data / "train.csv", target=args.target, training=True)
    test_set = RGB2TDataset(args.data / "test.csv", target=args.target, training=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )

    criterion: nn.Module = nn.L1Loss() if args.loss == "l1" else nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_rgb, fixed_target = next(iter(test_loader))
    fixed_rgb = fixed_rgb[:8].to(device)
    fixed_target = fixed_target[:8].to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for rgb, target in pbar:
            rgb = rgb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(rgb)
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if epoch > args.epochs // 2:
            decay = 1.0 - (epoch - args.epochs // 2) / (args.epochs - args.epochs // 2)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * decay

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                pred = model(fixed_rgb)
            rgb_vis = (fixed_rgb + 1) / 2
            target_vis = (fixed_target + 1) / 2 if args.target == "pseudo" else (fixed_target - fixed_target.min()) / (fixed_target.max() - fixed_target.min() + 1e-8)
            pred_vis = (pred + 1) / 2 if args.target == "pseudo" else (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            grid = torch.cat([rgb_vis, to_rgb_batch(target_vis), to_rgb_batch(pred_vis)], 0)
            save_image(grid, out / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

        if epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / f"checkpoint_epoch_{epoch}.pt")

        print(
            f"epoch {epoch}/{args.epochs} avg_loss={running / max(len(train_loader), 1):.4f}",
            flush=True,
        )

    torch.save({"model": model.state_dict(), "epoch": args.epochs}, out / "checkpoint.pt")
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
