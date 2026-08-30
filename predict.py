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
STATE_COLORS = {
    "lie": (255, 96, 32),
    "sit": (255, 200, 32),
    "other": (200, 32, 255),
    "off_bed": (32, 200, 32),
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
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="box stroke width in the saved visualization",
    )
    parser.add_argument("--font-scale", type=float, default=0.4)
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="include confidence in image labels (always retained in JSONL)",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--upscale",
        type=int,
        default=8,
        help="nearest-neighbor scale factor for saved visualizations only",
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


def render_prediction(
    image,
    detections: list[dict],
    upscale: int,
    line_width: int,
    font_scale: float,
    show_confidence: bool,
):
    """Upscale first, then draw thin boxes and compact labels at output resolution."""
    if upscale > 1:
        image = cv2.resize(
            image,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_NEAREST,
        )
    height, width = image.shape[:2]
    scale = float(upscale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value * scale)) for value in detection["bbox_xyxy"]]
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        state = detection["state"]
        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        label = state
        if show_confidence:
            label += f" {detection['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        label_x = max(0, min(x1, width - text_width - 6))
        label_top = y1 - text_height - baseline - 6
        if label_top < 0:
            label_top = y1
        label_bottom = min(height - 1, label_top + text_height + baseline + 6)
        cv2.rectangle(
            image,
            (label_x, label_top),
            (min(width - 1, label_x + text_width + 6), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (label_x + 3, label_top + text_height + 2),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return image


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

            annotated = render_prediction(
                result.orig_img.copy(),
                detections,
                args.upscale,
                args.line_width,
                args.font_scale,
                args.show_confidence,
            )
            rendered_name = f"{index:06d}_{source_path.parent.name}_{source_path.name}"
            if not cv2.imwrite(str(args.output / rendered_name), annotated):
                raise RuntimeError(f"Failed to write rendered prediction for {source_path}")

    print(f"images={len(images)} jsonl={jsonl_path} rendered_dir={args.output}")


if __name__ == "__main__":
    main()
