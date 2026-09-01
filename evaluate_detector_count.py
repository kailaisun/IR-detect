#!/usr/bin/env python3
"""Evaluate person count by counting class-agnostic detector boxes."""

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
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=project / "early_training/yolo26s_epoch11/weights/model.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project.parent / "yolo26_ir/dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "early_training/yolo26s_epoch11/counting",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-conf", type=float, default=0.01)
    parser.add_argument("--standard-conf", type=float, default=0.25)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_split(data_dir: Path, split: str) -> tuple[list[Path], np.ndarray]:
    paths = [Path(line) for line in (data_dir / f"{split}.txt").read_text().splitlines() if line]
    counts = []
    for path in paths:
        annotation = label_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        counts.append(sum(bool(line.strip()) for line in annotation.read_text().splitlines()))
    return paths, np.asarray(counts, dtype=np.int64)


def infer(
    model: YOLO,
    paths: list[Path],
    ground_truth: np.ndarray,
    args: argparse.Namespace,
    split: str,
) -> list[dict]:
    results = model.predict(
        source=str(args.data.resolve() / f"{split}.txt"),
        imgsz=args.imgsz,
        conf=args.inference_conf,
        iou=args.nms_iou,
        agnostic_nms=True,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
    )
    records = []
    output_path = args.output / f"{split}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, result in enumerate(results):
            detections = []
            if result.boxes is not None:
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
            record = {
                "index": index,
                "image": str(paths[index].relative_to(args.data.resolve() / "images" / split)),
                "ground_truth_count": int(ground_truth[index]),
                "detections": detections,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(records) != len(paths):
        raise RuntimeError(f"Expected {len(paths)} results for {split}, got {len(records)}")
    return records


def score_matrix(records: list[dict], max_det: int) -> np.ndarray:
    scores = np.zeros((len(records), max_det), dtype=np.float32)
    for row, record in enumerate(records):
        values = [item["confidence"] for item in record["detections"][:max_det]]
        scores[row, : len(values)] = values
    return scores


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


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
        "ground_truth_distribution": dict(sorted(Counter(map(int, ground_truth)).items())),
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
) -> tuple[float, dict, list[dict]]:
    thresholds = np.arange(
        args.inference_conf,
        0.9000001,
        args.threshold_step,
        dtype=np.float64,
    )
    rows = []
    best_rank = None
    best_threshold = None
    best_metrics = None
    for threshold in thresholds:
        result = metrics(ground_truth, predictions_at(scores, float(threshold)))
        row = {
            "threshold": round(float(threshold), 6),
            "mae": result["mae"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "exact_count_accuracy": result["exact_count_accuracy"],
            "mean_error": result["mean_error"],
        }
        rows.append(row)
        rank = (
            result["mae"],
            -result["exact_count_accuracy"],
            result["mse"],
            abs(result["mean_error"]),
            -float(threshold),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_metrics = result
    assert best_threshold is not None and best_metrics is not None
    return best_threshold, best_metrics, rows


def write_sweep(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


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
        title="YOLO26s epoch 11 count confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    if args.reuse_predictions:
        if not args.output.is_dir():
            raise FileNotFoundError(args.output)
        model = None
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        model = YOLO(str(args.model.resolve()))
    split_scores = {}
    split_ground_truth = {}
    for split in ("val", "test"):
        paths, ground_truth = read_split(args.data.resolve(), split)
        if args.reuse_predictions:
            records = read_predictions(args.output / f"{split}_predictions.jsonl")
            if len(records) != len(paths):
                raise RuntimeError(f"Expected {len(paths)} saved {split} predictions, got {len(records)}")
            saved_ground_truth = np.asarray(
                [record["ground_truth_count"] for record in records], dtype=np.int64
            )
            if not np.array_equal(saved_ground_truth, ground_truth):
                raise RuntimeError(f"Saved {split} ground truth does not match the current dataset")
        else:
            assert model is not None
            records = infer(model, paths, ground_truth, args, split)
        split_scores[split] = score_matrix(records, args.max_det)
        split_ground_truth[split] = ground_truth
        print(f"completed {split}: {len(records)} images")

    threshold, validation_metrics, sweep = select_threshold(
        split_ground_truth["val"], split_scores["val"], args
    )
    selected_test_prediction = predictions_at(split_scores["test"], threshold)
    selected_test_metrics = metrics(split_ground_truth["test"], selected_test_prediction)
    standard_test_prediction = predictions_at(split_scores["test"], args.standard_conf)
    standard_test_metrics = metrics(split_ground_truth["test"], standard_test_prediction)
    validation_majority_count = Counter(map(int, split_ground_truth["val"])).most_common(1)[0][0]
    majority_prediction = np.full_like(split_ground_truth["test"], validation_majority_count)
    majority_metrics = metrics(split_ground_truth["test"], majority_prediction)
    comparison = {
        "mae_absolute_reduction": majority_metrics["mae"] - selected_test_metrics["mae"],
        "mae_relative_reduction": (
            majority_metrics["mae"] - selected_test_metrics["mae"]
        ) / majority_metrics["mae"],
        "mse_absolute_reduction": majority_metrics["mse"] - selected_test_metrics["mse"],
        "mse_relative_reduction": (
            majority_metrics["mse"] - selected_test_metrics["mse"]
        ) / majority_metrics["mse"],
        "exact_count_accuracy_gain": (
            selected_test_metrics["exact_count_accuracy"]
            - majority_metrics["exact_count_accuracy"]
        ),
    }
    write_sweep(args.output / "validation_threshold_sweep.csv", sweep)
    plot_confusion(args.output / "count_confusion_matrix.png", selected_test_metrics)
    report = {
        "task": "person counting by the number of YOLO detection boxes",
        "model": str(args.model.resolve().relative_to(project)),
        "model_stage": "YOLO26s epoch 11 non-converged snapshot",
        "input": "80x62 infrared pseudo-color PNG only",
        "protocol": {
            "image_size": args.imgsz,
            "class_agnostic_nms": True,
            "nms_iou": args.nms_iou,
            "candidate_confidence": args.inference_conf,
            "max_detections": args.max_det,
            "threshold_selection": (
                "validation MAE minimum; ties resolved by exact-count accuracy, "
                "MSE, absolute bias, then the higher threshold"
            ),
            "test_set_used_for_threshold_selection": False,
        },
        "selected_confidence_threshold": round(threshold, 6),
        "validation_at_selected_threshold": validation_metrics,
        "test_at_selected_threshold": selected_test_metrics,
        "test_at_standard_confidence_0.25": standard_test_metrics,
        "test_majority_count_baseline": {
            "selection": "most frequent count in the validation split",
            "predicted_count": validation_majority_count,
            **majority_metrics,
        },
        "comparison_vs_majority_count_baseline": comparison,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
