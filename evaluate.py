#!/usr/bin/env python3
"""Evaluate a trained YOLO model on the scene-held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=project_dir / "weights/best.pt")
    parser.add_argument("--data", type=Path, default=project_dir / "dataset/dataset.yaml")
    parser.add_argument("--output", type=Path, default=project_dir / "results/metrics.json")
    parser.add_argument("--project", type=Path, default=project_dir / "results")
    parser.add_argument("--name", default="test_eval")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    model = YOLO(str(args.model.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    per_class = {}
    for class_id, class_name in metrics.names.items():
        per_class[class_name] = {
            "precision": float(metrics.box.p[class_id]),
            "recall": float(metrics.box.r[class_id]),
            "mAP50": float(metrics.box.ap50[class_id]),
            "mAP50-95": float(metrics.box.maps[class_id]),
        }
    report = {
        "split": "test",
        "rooms": ["room03", "room10", "room18"],
        "images": 12237,
        "instances": 16657,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "model": str(args.model),
        "imgsz": args.imgsz,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
