#!/usr/bin/env python3
"""Run infrared-only YOLO26 inference and emit boxes plus human states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


STATE_ZH = {
    "lie": "躺",
    "sit": "坐",
    "other": "其他行为",
    "off_bed": "床下",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="IR image or directory containing IR images")
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "weights/best.pt",
    )
    parser.add_argument("--output", type=Path, default=project_dir / "predictions")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="nearest-neighbor scale factor for saved visualizations",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source.resolve()]
    if source.is_dir():
        return sorted(
            path.resolve()
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(source)


def main() -> None:
    args = parse_args()
    images = collect_images(args.source)
    if not images:
        raise RuntimeError(f"No images found under {args.source}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Trained model not found: {args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output / "predictions.jsonl"
    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nms=True,
        agnostic_nms=True,
        device=args.device,
        stream=True,
        verbose=False,
    )

    with jsonl_path.open("w", encoding="utf-8") as output_jsonl:
        for index, result in enumerate(results):
            source_path = Path(result.path).resolve()
            detections = []
            if result.boxes is not None:
                for xyxy, confidence, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    strict=True,
                ):
                    state = result.names[int(class_id)]
                    detections.append(
                        {
                            "bbox_xyxy": [round(value, 3) for value in xyxy],
                            "confidence": round(float(confidence), 6),
                            "class_id": int(class_id),
                            "state": state,
                            "state_zh": STATE_ZH.get(state, state),
                        }
                    )
            # Keep exported JSON portable and avoid leaking machine-specific paths.
            record = {"image": source_path.name, "detections": detections}
            output_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            annotated = result.plot()
            if args.upscale > 1:
                annotated = cv2.resize(
                    annotated,
                    None,
                    fx=args.upscale,
                    fy=args.upscale,
                    interpolation=cv2.INTER_NEAREST,
                )
            rendered_name = f"{index:06d}_{source_path.parent.name}_{source_path.name}"
            if not cv2.imwrite(str(args.output / rendered_name), annotated):
                raise RuntimeError(f"Failed to write rendered prediction for {source_path}")

    print(f"images={len(images)} jsonl={jsonl_path} rendered_dir={args.output}")


if __name__ == "__main__":
    main()
