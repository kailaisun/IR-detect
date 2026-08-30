#!/usr/bin/env python3
"""Strict batch-one MMDetection latency benchmark on preloaded IR frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("mmdetection_data/images/test"))
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.images.rglob("*.png"))
    if len(paths) < args.frames:
        raise RuntimeError(f"Found only {len(paths)} images")
    indices = np.linspace(0, len(paths) - 1, args.frames, dtype=int)
    frames = [cv2.imread(str(paths[index]), cv2.IMREAD_COLOR) for index in indices]
    if any(frame is None for frame in frames):
        raise RuntimeError("Failed to load one or more benchmark frames")

    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    for frame in frames[: args.warmup]:
        inference_detector(model, frame)
    torch.cuda.synchronize()

    timings = []
    for _ in range(args.repeats):
        for frame in frames:
            torch.cuda.synchronize()
            start = time.perf_counter()
            inference_detector(model, frame)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(timings)
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "input": "preloaded 80x62 three-channel IR pseudo-color ndarray",
        "model_input_size": 320,
        "batch_size": 1,
        "frames_per_repeat": args.frames,
        "repeats": args.repeats,
        "warmup_frames": args.warmup,
        "latency_ms_mean": float(values.mean()),
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "fps_from_mean": float(1000.0 / values.mean()),
        "scope": "MMDetection ndarray pipeline + forward + postprocess; disk I/O excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
