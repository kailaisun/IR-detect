#!/usr/bin/env python3
"""Train Pix2Pix for RGB -> thermal-field or pseudo-color."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import RGB2TDataset
from pix2pix import NLayerDiscriminator, init_weights, make_generator


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["thermal", "pseudo"], default="thermal")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "runs" / "pix2pix")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
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
    generator = make_generator(out_channels, activation).to(device)
    discriminator = NLayerDiscriminator(in_channels=3 + out_channels).to(device)
    init_weights(generator)
    init_weights(discriminator)

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

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_rgb, fixed_target = next(iter(test_loader))
    fixed_rgb = fixed_rgb[:8].to(device)
    fixed_target = fixed_target[:8].to(device)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for i, (rgb, target) in enumerate(pbar):
            rgb = rgb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            fake = generator(rgb)
            pred_real = discriminator(torch.cat([rgb, target], 1))
            pred_fake = discriminator(torch.cat([rgb, fake.detach()], 1))
            loss_d = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real))
                + criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            pred_fake = discriminator(torch.cat([rgb, fake], 1))
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake, target) * args.lambda_l1
            loss_g = loss_g_gan + loss_g_l1
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()

            if i % 100 == 0:
                pbar.set_postfix(
                    d=f"{loss_d.item():.3f}",
                    g_gan=f"{loss_g_gan.item():.3f}",
                    l1=f"{loss_g_l1.item():.3f}",
                )

        if epoch > args.epochs // 2:
            decay = 1.0 - (epoch - args.epochs // 2) / (args.epochs - args.epochs // 2)
            for opt in (optimizer_g, optimizer_d):
                for group in opt.param_groups:
                    group["lr"] = args.lr * decay

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            generator.eval()
            with torch.inference_mode():
                fake = generator(fixed_rgb)
            rgb_vis = (fixed_rgb + 1) / 2
            target_vis = (fixed_target + 1) / 2 if args.target == "pseudo" else (fixed_target - fixed_target.min()) / (fixed_target.max() - fixed_target.min() + 1e-8)
            fake_vis = (fake + 1) / 2 if args.target == "pseudo" else (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
            grid = torch.cat([rgb_vis, to_rgb_batch(target_vis), to_rgb_batch(fake_vis)], 0)
            save_image(grid, out / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

        if epoch % args.save_every == 0:
            torch.save(
                {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": epoch},
                out / f"checkpoint_epoch_{epoch}.pt",
            )

    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "epoch": args.epochs},
        out / "checkpoint.pt",
    )
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
