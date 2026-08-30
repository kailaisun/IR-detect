#!/usr/bin/env python3
"""Render compact MMDetection IR predictions with paired RGB references."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from mmdet.apis import inference_detector, init_detector


COLORS = ((56, 142, 255), (80, 200, 120), (220, 170, 60), (210, 90, 200))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("examples/pairs.json"))
    parser.add_argument("--source", type=Path, default=Path("../data"))
    parser.add_argument("--rgb-source", type=Path, default=Path("examples"))
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--scale", type=int, default=8)
    return parser.parse_args()


def find_ir(source: Path, pair: dict) -> Path:
    session_root = source / pair["scene"] / pair["session"]
    canonical = session_root / "ir" / f"{pair['ir_timestamp_ms']}.png"
    if canonical.is_file():
        return canonical
    matches = list((session_root / "ir").glob(f"*{pair['ir_timestamp_ms']}*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one IR image for {pair}, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(args.pairs.read_text())
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    classes = model.dataset_meta["classes"]
    records = []

    for pair in pairs:
        ir_path = find_ir(args.source, pair)
        image = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        result = inference_detector(model, image)
        instances = result.pred_instances.cpu()
        canvas = cv2.resize(
            image,
            (image.shape[1] * args.scale, image.shape[0] * args.scale),
            interpolation=cv2.INTER_NEAREST,
        )
        detections = []
        for box, score, label in zip(
            instances.bboxes.numpy(), instances.scores.numpy(), instances.labels.numpy()
        ):
            if float(score) < args.conf:
                continue
            class_id = int(label)
            x1, y1, x2, y2 = [int(round(value * args.scale)) for value in box]
            color = COLORS[class_id]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = classes[class_id]
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )
            text_y = max(text_height + 2, y1 - 3)
            cv2.rectangle(
                canvas,
                (x1, text_y - text_height - 2),
                (x1 + text_width + 3, text_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                text,
                (x1 + 1, text_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "bbox_xyxy": [float(value) for value in box],
                    "confidence": float(score),
                    "class_id": class_id,
                    "state": classes[class_id],
                }
            )

        prediction_name = f"example_{pair['state']}.png"
        cv2.imwrite(str(args.output / prediction_name), canvas)
        shutil.copy2(
            args.rgb_source / pair["rgb_reference"],
            args.output / pair["rgb_reference"],
        )
        updated_pair = dict(pair)
        updated_pair["ir_prediction"] = prediction_name
        records.append({"pair": updated_pair, "ir_source": str(ir_path), "detections": detections})

    (args.output / "pairs.json").write_text(
        json.dumps([record["pair"] for record in records], indent=2) + "\n"
    )
    with (args.output / "predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
