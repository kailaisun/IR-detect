#!/usr/bin/env python3
"""Plot compact training-loss and validation-mAP curves from MMEngine JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalars", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scalars.read_text().splitlines()]
    train = [row for row in rows if "loss" in row and "iter" in row]
    val = [row for row in rows if "coco/bbox_mAP" in row]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["step"] for row in train], [row["loss"] for row in train])
    axes[0].set(title="Training loss", xlabel="Iteration", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    if val:
        axes[1].plot(
            [row["step"] for row in val],
            [row["coco/bbox_mAP"] for row in val],
            marker="o",
            label="mAP50-95",
        )
        axes[1].plot(
            [row["step"] for row in val],
            [row["coco/bbox_mAP_50"] for row in val],
            marker="o",
            label="mAP50",
        )
        axes[1].legend()
    axes[1].set(title="Validation accuracy", xlabel="Iteration", ylabel="mAP")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
