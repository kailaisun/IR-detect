#!/usr/bin/env python3
"""Train Pix2Pix for IR pseudo-color -> RGB translation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pix2pix import (
    NLayerDiscriminator,
    PairedDataset,
    UNetGenerator,
    init_weights,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "pix2pix")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def update_lr(optimizer: torch.optim.Optimizer, decay: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * decay


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "samples").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_set = PairedDataset(args.data / "train.csv", training=True)
    test_set = PairedDataset(args.data / "test.csv", training=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    generator = UNetGenerator(in_channels=3, out_channels=3, num_downs=6).to(device)
    discriminator = NLayerDiscriminator(in_channels=6, n_layers=3).to(device)
    init_weights(generator)
    init_weights(discriminator)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_ir, fixed_rgb = next(iter(test_loader))
    fixed_ir, fixed_rgb = fixed_ir[:8].to(device), fixed_rgb[:8].to(device)

    for epoch in range(1, args.epochs + 1):
        for i, (ir, rgb) in enumerate(train_loader):
            ir = ir.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)

            # ---- discriminator ----
            fake = generator(ir)
            fake_detached = fake.detach()
            pred_real = discriminator(torch.cat([ir, rgb], 1))
            pred_fake = discriminator(torch.cat([ir, fake_detached], 1))
            loss_d = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real))
                + criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            # ---- generator ----
            pred_fake = discriminator(torch.cat([ir, fake], 1))
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake, rgb) * args.lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(
                    f"epoch {epoch}/{args.epochs} iter {i} "
                    f"loss_D={loss_d.item():.4f} loss_G_GAN={loss_g_gan.item():.4f} "
                    f"loss_G_L1={loss_g_l1.item():.4f}",
                    flush=True,
                )

        if epoch > args.epochs // 2:
            update_lr(optimizer_g, 1.0 - (epoch - args.epochs // 2) / (args.epochs - args.epochs // 2))
            update_lr(optimizer_d, 1.0 - (epoch - args.epochs // 2) / (args.epochs - args.epochs // 2))

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.inference_mode():
                sample_fake = generator(fixed_ir)
            sample_fake = sample_fake[:8]
            grid = torch.cat([fixed_ir[:8], sample_fake, fixed_rgb[:8]], 0)
            grid = (grid + 1) / 2
            save_image(grid, args.output / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
            torch.save(
                {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": epoch},
                args.output / "checkpoint.pt",
            )

        if epoch % args.save_every == 0:
            torch.save(
                {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": epoch},
                args.output / f"checkpoint_epoch_{epoch}.pt",
            )

    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": args.epochs},
        args.output / "checkpoint.pt",
    )
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
