#!/usr/bin/env python3
"""Train weighted ResNet18 for one-person vs two-people IR classification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from common import CLASS_NAMES, DEFAULT_MODEL, build_loader, build_model, evaluate_model, set_seed


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "dataset")
    parser.add_argument("--dataset-root", type=Path, default=root.parents[1] / "data")
    parser.add_argument("--output", type=Path, default=root / "runs/resnet18_weighted")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    class_weights = torch.tensor(
        [manifest["train_class_weights"][name] for name in CLASS_NAMES],
        dtype=torch.float32,
    )
    if args.unweighted:
        class_weights = torch.ones_like(class_weights)
    device = torch.device(args.device)
    train_loader = build_loader(
        args.data / "train.csv", args.dataset_root, args.image_size,
        args.batch_size, args.workers, True, args.cache,
    )
    val_loader = build_loader(
        args.data / "val.csv", args.dataset_root, args.image_size,
        args.batch_size, args.workers, False, args.cache,
    )
    model = build_model(
        args.model, pretrained=not args.no_pretrained, img_size=args.image_size
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss) * len(labels)
            sample_count += len(labels)
        scheduler.step()
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items() if not isinstance(value, (dict, list))},
        }
        history.append(row)
        checkpoint = {
            "model_state": model.state_dict(),
            "model": args.model,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "epoch": epoch,
            "weighted_loss": not args.unweighted,
            "class_weights": class_weights.tolist(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} accuracy={val_metrics['accuracy']:.4f} "
            f"balanced_accuracy={val_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best macro_f1={best_f1:.4f}")
            break
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="accuracy")
    axes[1].plot(epochs, [row["val_balanced_accuracy"] for row in history], label="balanced accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="macro F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)


if __name__ == "__main__":
    main()
