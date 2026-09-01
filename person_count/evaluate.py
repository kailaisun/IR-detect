#!/usr/bin/env python3
"""Evaluate a ResNet person-count checkpoint on the held-out test rooms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CLASS_NAMES, build_loader, build_model, evaluate_model


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/resnet18_weighted/best.pt")
    parser.add_argument("--data", type=Path, default=root / "dataset")
    parser.add_argument("--dataset-root", type=Path, default=root.parents[1] / "data")
    parser.add_argument("--output", type=Path, default=root / "results/metrics.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = labels[order]
    true_positive = np.cumsum(labels == 1)
    false_positive = np.cumsum(labels == 0)
    tpr = np.concatenate(([0.0], true_positive / max(true_positive[-1], 1)))
    fpr = np.concatenate(([0.0], false_positive / max(false_positive[-1], 1)))
    return fpr, tpr


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    loader = build_loader(
        args.data / "test.csv", args.dataset_root, checkpoint["image_size"],
        args.batch_size, args.workers, False, args.cache,
    )
    metrics, labels, _, probabilities = evaluate_model(model, loader, device)
    manifest = json.loads((args.data / "manifest.json").read_text())
    report = {
        "task": "one-person vs two-people infrared image classification",
        "split": "test",
        "rooms": manifest["split_rooms"]["test"],
        "images": len(labels),
        "model": "resnet18",
        "checkpoint": str(args.checkpoint),
        "image_size": checkpoint["image_size"],
        "weighted_loss": checkpoint["weighted_loss"],
        "class_weights": dict(zip(CLASS_NAMES, checkpoint["class_weights"], strict=True)),
        "best_epoch": checkpoint["epoch"],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    confusion = np.asarray(metrics["confusion_matrix"])
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    for matrix, name, title in [
        (confusion, "confusion_matrix.png", "Confusion matrix"),
        (normalized, "confusion_matrix_normalized.png", "Normalized confusion matrix"),
    ]:
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if matrix is normalized else None)
        for row in range(2):
            for column in range(2):
                value = f"{matrix[row, column]:.3f}" if matrix is normalized else str(matrix[row, column])
                axis.text(column, row, value, ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True", title=title)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(args.output.parent / name, dpi=180)
        plt.close(figure)
    fpr, tpr = roc_curve(labels, probabilities[:, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"AUC={metrics['roc_auc_two_people']:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Two-people ROC")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output.parent / "roc_curve.png", dpi=180)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
