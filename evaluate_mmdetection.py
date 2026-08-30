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
    return {
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
