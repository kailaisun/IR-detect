#!/usr/bin/env python3
"""Train a YOLO detector to predict a person bounding box and human state."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Avoid Anaconda MKL/libgomp conflicts in Ultralytics DDP subprocesses.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--data", type=Path, default=project_dir / "dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--name", default="yolo26s_ir_status")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str((project_dir / "runs").resolve()),
        name=args.name,
        exist_ok=False,
        cache=False if args.no_cache else "ram",
        fraction=args.fraction,
        val=not args.no_val,
        patience=20,
        plots=True,
        save_period=10,
        pretrained=True,
        optimizer="auto",
        cls_pw=0.25,
        # Preserve the physical meaning of the thermal palette.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        translate=0.10,
        scale=0.30,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
