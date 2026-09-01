#!/usr/bin/env python3
"""Evaluate YOLO box-counting on the exact ResNet one/two-person splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


COUNT_LABELS = ("0", "1", "2", "3+")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    project = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=project / "early_training/yolo26s_epoch11/weights/model.pt",
    )
    parser.add_argument("--data", type=Path, default=root / "dataset")
    parser.add_argument("--dataset-root", type=Path, default=project.parent / "data")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project / "early_training/yolo26s_epoch11/counting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/yolo26s_epoch11_metrics.json",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-conf", type=float, default=0.01)
    parser.add_argument("--standard-conf", type=float, default=0.25)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    return parser.parse_args()


def detection_key(classification_path: str) -> str:
    return classification_path.replace("/ir/", "/")


def read_classification_split(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ground_truth_count"] = int(row["label"]) + 1
        row["detection_key"] = detection_key(row["image"])
    return rows


def read_detector_predictions(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return {record["image"]: record for record in records}


def serialize_detections(result) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections
    for box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        strict=True,
    ):
        detections.append(
            {
                "bbox_xyxy": [round(float(value), 3) for value in box],
                "confidence": round(float(confidence), 6),
                "class_id": int(class_id),
                "state": result.names[int(class_id)],
            }
        )
    return detections


def complete_predictions(
    model: YOLO,
    rows: list[dict],
    existing: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], int]:
    missing = [row for row in rows if row["detection_key"] not in existing]
    inferred = {}
    if missing:
        results = model.predict(
            source=[str(args.dataset_root / row["image"]) for row in missing],
            imgsz=args.imgsz,
            conf=args.inference_conf,
            iou=args.nms_iou,
            agnostic_nms=True,
            max_det=args.max_det,
            device=args.device,
            stream=True,
            verbose=False,
        )
        for row, result in zip(missing, results, strict=True):
            inferred[row["detection_key"]] = {
                "image": row["detection_key"],
                "detections": serialize_detections(result),
            }
    combined = []
    annotation_mismatches = 0
    for row in rows:
        record = existing.get(row["detection_key"], inferred.get(row["detection_key"]))
        if record is None:
            raise RuntimeError(f"Missing YOLO prediction for {row['image']}")
        if "ground_truth_count" in record and record["ground_truth_count"] != row["ground_truth_count"]:
            annotation_mismatches += 1
        combined.append(
            {
                "image": row["image"],
                "frame_id": row["frame_id"],
                "ground_truth_count": row["ground_truth_count"],
                "detections": record["detections"],
                "source": "reused" if row["detection_key"] in existing else "newly_inferred",
            }
        )
    return combined, list(inferred.values()), annotation_mismatches


def score_matrix(records: list[dict], max_det: int) -> np.ndarray:
    scores = np.zeros((len(records), max_det), dtype=np.float32)
    for row, record in enumerate(records):
        values = [item["confidence"] for item in record["detections"][:max_det]]
        scores[row, : len(values)] = values
    return scores


def count_bin(value: int) -> int:
    return min(value, 3)


def metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction.astype(np.float64) - ground_truth
    absolute_error = np.abs(error)
    squared_error = error**2
    confusion = np.zeros((4, 4), dtype=np.int64)
    for target, predicted in zip(ground_truth, prediction, strict=True):
        confusion[count_bin(int(target)), count_bin(int(predicted))] += 1
    per_count = {}
    for value in sorted(np.unique(ground_truth)):
        mask = ground_truth == value
        per_count[str(int(value))] = {
            "images": int(mask.sum()),
            "mae": float(absolute_error[mask].mean()),
            "mse": float(squared_error[mask].mean()),
            "exact_accuracy": float((error[mask] == 0).mean()),
            "mean_error": float(error[mask].mean()),
        }
    return {
        "images": int(len(ground_truth)),
        "mae": float(absolute_error.mean()),
        "mse": float(squared_error.mean()),
        "rmse": float(math.sqrt(squared_error.mean())),
        "exact_count_accuracy": float((error == 0).mean()),
        "within_one_accuracy": float((absolute_error <= 1).mean()),
        "mean_error": float(error.mean()),
        "under_count_rate": float((error < 0).mean()),
        "over_count_rate": float((error > 0).mean()),
        "prediction_distribution": dict(sorted(Counter(map(int, prediction)).items())),
        "confusion_matrix_labels": COUNT_LABELS,
        "confusion_matrix": confusion.tolist(),
        "per_ground_truth_count": per_count,
    }


def predictions_at(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).sum(axis=1).astype(np.int64)


def select_threshold(
    ground_truth: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    best_rank = None
    selected = None
    for threshold in np.arange(
        args.inference_conf,
        0.9000001,
        args.threshold_step,
        dtype=np.float64,
    ):
        result = metrics(ground_truth, predictions_at(scores, float(threshold)))
        rank = (
            result["mae"],
            -result["exact_count_accuracy"],
            result["mse"],
            abs(result["mean_error"]),
            -float(threshold),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            selected = (float(threshold), result)
    assert selected is not None
    return selected


def plot_confusion(path: Path, payload: dict) -> None:
    matrix = np.asarray(payload["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5.4, 4.7))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=COUNT_LABELS,
        yticklabels=COUNT_LABELS,
        xlabel="Predicted person count",
        ylabel="Ground-truth person count",
        title="YOLO26s epoch 11 on the ResNet test subset",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    payload = {}
    for split in ("val", "test"):
        rows = read_classification_split(args.data / f"{split}.csv")
        existing = read_detector_predictions(args.predictions / f"{split}_predictions.jsonl")
        records, inferred, mismatches = complete_predictions(model, rows, existing, args)
        with (args.output.parent / f"yolo26s_epoch11_{split}_missing_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record in inferred:
                handle.write(json.dumps(record) + "\n")
        payload[split] = {
            "rows": rows,
            "records": records,
            "scores": score_matrix(records, args.max_det),
            "ground_truth": np.asarray(
                [row["ground_truth_count"] for row in rows], dtype=np.int64
            ),
            "reused_predictions": len(rows) - len(inferred),
            "newly_inferred_predictions": len(inferred),
            "count_label_vs_detection_annotation_mismatches": mismatches,
        }
    threshold, validation_metrics = select_threshold(
        payload["val"]["ground_truth"], payload["val"]["scores"], args
    )
    test_metrics = metrics(
        payload["test"]["ground_truth"],
        predictions_at(payload["test"]["scores"], threshold),
    )
    standard_metrics = metrics(
        payload["test"]["ground_truth"],
        predictions_at(payload["test"]["scores"], args.standard_conf),
    )
    report = {
        "task": "person counting by YOLO box count on the ResNet one/two-person split",
        "model": "early_training/yolo26s_epoch11/weights/model.pt",
        "protocol": {
            "image_size": args.imgsz,
            "class_agnostic_nms": True,
            "nms_iou": args.nms_iou,
            "threshold_selected_on_validation_only": True,
            "ground_truth": "person_count CSV label, shared with the ResNet classifier",
        },
        "selected_confidence_threshold": round(threshold, 6),
        "data_reuse": {
            split: {
                key: payload[split][key]
                for key in (
                    "reused_predictions",
                    "newly_inferred_predictions",
                    "count_label_vs_detection_annotation_mismatches",
                )
            }
            for split in ("val", "test")
        },
        "validation_at_selected_threshold": validation_metrics,
        "test_at_selected_threshold": test_metrics,
        "test_at_standard_confidence_0.25": standard_metrics,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    plot_confusion(args.output.parent / "yolo26s_epoch11_confusion_matrix.png", test_metrics)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
