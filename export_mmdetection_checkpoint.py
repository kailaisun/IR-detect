#!/usr/bin/env python3
"""Export inference-only FP16 checkpoint small enough for ordinary GitHub."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu")
    state_dict = {
        key: value.half() if torch.is_floating_point(value) else value
        for key, value in checkpoint["state_dict"].items()
    }
    payload = {
        "meta": checkpoint.get("meta", {}),
        "state_dict": state_dict,
        "message": "Inference-only FP16 storage; MMDetection loads it into FP32 modules.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"{args.output}: {args.output.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
