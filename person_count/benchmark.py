#!/usr/bin/env python3
"""Benchmark batch-one ResNet person-count latency on preloaded IR frames."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import build_model, build_transform


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/resnet18_weighted/best.pt")
    parser.add_argument("--data", type=Path, default=root / "dataset/test.csv")
    parser.add_argument("--dataset-root", type=Path, default=root.parents[1] / "data")
    parser.add_argument("--output", type=Path, default=root / "results/benchmark.json")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    with args.data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indices = np.linspace(0, len(rows) - 1, args.frames, dtype=int)
    frames = []
    for index in indices:
        with Image.open(args.dataset_root / rows[index]["image"]) as image:
            frames.append(np.asarray(image.convert("RGB")).copy())
    device = torch.device(args.device)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    transform = build_transform(checkpoint["image_size"], training=False)

    @torch.inference_mode()
    def infer(frame: np.ndarray) -> None:
        tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        model(tensor).softmax(dim=1)

    for frame in frames[: args.warmup]:
        infer(frame)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.repeats):
        for frame in frames:
            torch.cuda.synchronize()
            start = time.perf_counter()
            infer(frame)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings)
    report = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "input": "preloaded 80x62 three-channel IR pseudo-color image",
        "model_input_size": checkpoint["image_size"],
        "batch_size": 1,
        "frames_per_repeat": args.frames,
        "repeats": args.repeats,
        "warmup_frames": args.warmup,
        "latency_ms_mean": float(values.mean()),
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "fps_from_mean": float(1000.0 / values.mean()),
        "scope": "letterbox preprocessing + host-to-device transfer + ResNet18 forward + softmax; disk I/O excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
