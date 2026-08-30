#!/usr/bin/env python3
"""Evaluate one MMDetection checkpoint on the held-out IR test rooms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def mean_valid(values: np.ndarray) -> float:
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def precision_recall_operating_point(
    evaluator: COCOeval,
    coco_gt: COCO,
    iou_threshold: float = 0.5,
) -> dict:
    """Select one confidence threshold maximizing macro F1 at a fixed IoU."""
    iou_index = int(np.abs(evaluator.params.iouThrs - iou_threshold).argmin())
    area_range = tuple(evaluator.params.areaRng[0])
    max_detections = evaluator.params.maxDets[-1]
    curves = {}
    for category_id in coco_gt.getCatIds():
        records = [
            record
            for record in evaluator.evalImgs
            if record is not None
            and record["category_id"] == category_id
            and tuple(record["aRng"]) == area_range
            and record["maxDet"] == max_detections
        ]
        scores = np.concatenate(
            [np.asarray(record["dtScores"], dtype=np.float64) for record in records]
        )
        matches = np.concatenate(
            [np.asarray(record["dtMatches"][iou_index]) for record in records]
        )
        ignored = np.concatenate(
            [np.asarray(record["dtIgnore"][iou_index], dtype=bool) for record in records]
        )
        targets = int(
            sum(np.count_nonzero(~np.asarray(record["gtIgnore"], dtype=bool)) for record in records)
        )
        order = np.argsort(-scores, kind="mergesort")
        scores = scores[order]
        valid = ~ignored[order]
        true_positives = (matches[order] > 0) & valid
        false_positives = (matches[order] == 0) & valid
        curves[category_id] = {
            "scores": scores,
            "tp": np.cumsum(true_positives),
            "fp": np.cumsum(false_positives),
            "targets": targets,
        }

    thresholds = np.linspace(0.0, 1.0, 1001)
    macro_f1 = []
    threshold_metrics = []
    for threshold in thresholds:
        class_metrics = {}
        for category_id, curve in curves.items():
            count = int(np.searchsorted(-curve["scores"], -threshold, side="right"))
            true_positive = float(curve["tp"][count - 1]) if count else 0.0
            false_positive = float(curve["fp"][count - 1]) if count else 0.0
            false_negative = float(curve["targets"] - true_positive)
            precision = true_positive / max(true_positive + false_positive, 1.0)
            recall = true_positive / max(true_positive + false_negative, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
            class_metrics[category_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
            }
        threshold_metrics.append(class_metrics)
        macro_f1.append(np.mean([metrics["f1"] for metrics in class_metrics.values()]))

    best_index = int(np.argmax(macro_f1))
    selected = threshold_metrics[best_index]
    per_class = {
        coco_gt.cats[category_id]["name"]: metrics
        for category_id, metrics in selected.items()
    }
    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": float(thresholds[best_index]),
        "selection": "single confidence threshold maximizing macro class F1",
        "precision": float(np.mean([metrics["precision"] for metrics in selected.values()])),
        "recall": float(np.mean([metrics["recall"] for metrics in selected.values()])),
        "f1": float(macro_f1[best_index]),
        "per_class": per_class,
    }


def detailed_coco_metrics(gt_path: Path, prediction_path: Path) -> dict:
    coco_gt = COCO(str(gt_path))
    predictions = json.loads(prediction_path.read_text())
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    per_class = {}
    for class_index, category_id in enumerate(evaluator.params.catIds):
        name = coco_gt.cats[category_id]["name"]
        per_class[name] = {
            "map50_95": mean_valid(precision[:, :, class_index, 0, -1]),
            "map50": mean_valid(precision[0, :, class_index, 0, -1]),
            "max_recall50": float(recall[0, class_index, 0, -1]),
        }
    operating_point = precision_recall_operating_point(evaluator, coco_gt)
    for name, values in operating_point["per_class"].items():
        per_class[name].update(values)
    return {
        "precision": operating_point["precision"],
        "recall": operating_point["recall"],
        "f1": operating_point["f1"],
        "operating_point": {
            key: value for key, value in operating_point.items() if key != "per_class"
        },
        "map50_95": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "ar100": float(evaluator.stats[8]),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_prefix = args.output.parent / "test_predictions"

    cfg = Config.fromfile(args.config)
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.work_dir = str(args.output.parent / "test_work_dir")
    cfg.test_dataloader.batch_size = args.batch
    cfg.test_evaluator.outfile_prefix = str(prediction_prefix)
    runner = Runner.from_cfg(cfg)
    native_metrics = runner.test()

    gt_path = Path(cfg.test_evaluator.ann_file)
    prediction_path = Path(str(prediction_prefix) + ".bbox.json")
    detailed = detailed_coco_metrics(gt_path, prediction_path)
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "image_modality": "infrared pseudo-color PNG only; RGB excluded",
        "test_images": len(COCO(str(gt_path)).imgs),
        "native_mmdetection_metrics": native_metrics,
        "metrics": detailed,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
