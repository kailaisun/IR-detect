#!/usr/bin/env python3
"""Build scene-disjoint IR manifests for one-person vs two-people classification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


TRAIN_ROOMS = {
    "room01", "room02", "room04", "room05", "room06", "room08", "room09",
    "room11", "room12", "room14", "room15", "room16", "room17", "room19",
}
VAL_ROOMS = {"room07", "room13", "room20"}
TEST_ROOMS = {"room03", "room10", "room18"}
CLASS_NAMES = ("one_person", "two_people")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dataset")
    return parser.parse_args()


def split_for_scene(scene_id: str) -> str:
    if scene_id in TRAIN_ROOMS:
        return "train"
    if scene_id in VAL_ROOMS:
        return "val"
    if scene_id in TEST_ROOMS:
        return "test"
    raise ValueError(f"Unknown scene: {scene_id}")


def count_label(record: dict) -> tuple[int | None, str]:
    flags = record.get("flags", {})
    if not flags.get("labeled", False):
        return None, "unlabeled"
    if flags.get("discarded", False):
        return None, "discarded"
    if flags.get("high_noise", False):
        return None, "high_noise"
    values = {
        obj.get("parsed", {}).get("n_person", "")
        for obj in record.get("objects", [])
        if obj.get("parsed", {}).get("n_person", "")
    }
    counts = set()
    for value in values:
        if value.startswith("0_"):
            counts.add(0)
        elif value.startswith("1_"):
            counts.add(1)
        elif value.startswith("2_"):
            counts.add(2)
        elif value.startswith("3_"):
            counts.add(3)
    if len(counts) > 1:
        return None, "conflicting_count"
    if not counts:
        return None, "missing_count"
    count = next(iter(counts))
    if count not in (1, 2):
        return None, f"excluded_count_{count}"
    return count - 1, ""


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    counts = defaultdict(Counter)
    skipped = Counter()
    for labels_path in sorted((source / "data").glob("room*/**/labels.jsonl")):
        with labels_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                label, reason = count_label(record)
                if label is None:
                    skipped[reason] += 1
                    continue
                split = split_for_scene(record["scene_id"])
                image_path = labels_path.parent / record["ir"]
                rows[split].append(
                    {
                        "image": image_path.relative_to(source / "data").as_posix(),
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                        "frame_id": record["frame_id"],
                        "scene_id": record["scene_id"],
                        "session_id": record["session_id"],
                    }
                )
                counts[split][label] += 1
    for split in ("train", "val", "test"):
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("image", "label", "class_name", "frame_id", "scene_id", "session_id"),
            )
            writer.writeheader()
            writer.writerows(rows[split])
    train_total = sum(counts["train"].values())
    class_weights = [
        train_total / (len(CLASS_NAMES) * counts["train"][class_id])
        for class_id in range(len(CLASS_NAMES))
    ]
    manifest = {
        "source": str(source),
        "image_modality": "infrared pseudo-color PNG only; RGB excluded",
        "classes": list(CLASS_NAMES),
        "split_rooms": {
            "train": sorted(TRAIN_ROOMS),
            "val": sorted(VAL_ROOMS),
            "test": sorted(TEST_ROOMS),
        },
        "splits": {
            split: {
                "images": len(rows[split]),
                "class_counts": {
                    CLASS_NAMES[class_id]: counts[split][class_id]
                    for class_id in range(len(CLASS_NAMES))
                },
                "two_people_ratio": counts[split][1] / len(rows[split]),
            }
            for split in ("train", "val", "test")
        },
        "train_class_weights": dict(zip(CLASS_NAMES, class_weights, strict=True)),
        "skipped": dict(skipped),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
