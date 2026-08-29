#!/usr/bin/env python3
"""Convert household_ir labels.jsonl files to a scene-disjoint YOLO dataset.

Only infrared PNG files are linked. RGB files are never read or referenced.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


STATUS_NAMES = {
    0: "lie",
    1: "sit",
    2: "other",
    3: "off_bed",
}
STATUS_NAMES_ZH = {
    "lie": "躺",
    "sit": "坐",
    "other": "其他行为",
    "off_bed": "床下",
}
STATUS_TO_ID = {name: class_id for class_id, name in STATUS_NAMES.items()}

# Each validation/test split covers numbered, script_b1, and script_b2 data.
VAL_SCENES = {"room07", "room13", "room20"}
TEST_SCENES = {"room03", "room10", "room18"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_dir.parent,
        help="household_ir dataset_v1 root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "dataset",
        help="output YOLO dataset directory",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy IR PNGs instead of creating relative symlinks",
    )
    parser.add_argument(
        "--lists-only",
        action="store_true",
        help="refresh manifests/lists without rewriting existing links and labels",
    )
    return parser.parse_args()


def split_for_scene(scene_id: str) -> str:
    if scene_id in VAL_SCENES:
        return "val"
    if scene_id in TEST_SCENES:
        return "test"
    return "train"


def object_status(obj: dict) -> str | None:
    parsed = obj.get("parsed") or {}
    posture = parsed.get("posture_canon", "")
    if posture in {"lie", "sit", "other"}:
        return posture
    # Reclining has only 69 boxes in the full dataset and is not shared by all
    # batches, so follow posture_canon semantics and fold it into other.
    if posture == "unmapped":
        return "other"
    if parsed.get("place") == "0_人在床下":
        return "off_bed"
    return None


def yolo_line(obj: dict) -> tuple[str | None, str]:
    """Return (YOLO line, skip reason)."""
    parsed = obj.get("parsed") or {}
    if parsed.get("high_noise") or parsed.get("n_person") == "高噪音图像":
        return None, "object_high_noise"
    if parsed.get("n_person") == "0_无人":
        return None, "no_person_marker"
    if obj.get("bbox_status", ""):
        return None, f"bbox_{obj['bbox_status']}"

    bbox = obj.get("bbox_xyxy")
    canvas = obj.get("canvas") or [80, 62]
    if not bbox or len(bbox) != 4 or len(canvas) != 2:
        return None, "bbox_missing"
    width, height = map(float, canvas)
    if width <= 0 or height <= 0:
        return None, "canvas_invalid"

    x1, y1, x2, y2 = map(float, bbox)
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None, "bbox_degenerate_after_clip"

    state = object_status(obj)
    if state is None:
        return None, "state_unknown"
    class_id = STATUS_TO_ID[state]
    x_center = (x1 + x2) / (2.0 * width)
    y_center = (y1 + y2) / (2.0 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    line = f"{class_id} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
    return line, ""


def link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve() and not copy_images:
            return
        destination.unlink()
    elif destination.exists():
        if copy_images:
            return
        raise FileExistsError(f"Refusing to replace non-symlink: {destination}")

    if copy_images:
        import shutil

        shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def write_yaml(output: Path) -> None:
    lines = [
        f"path: {output.resolve()}",
        "train: train.txt",
        "val: val.txt",
        "test: test.txt",
        "names:",
    ]
    lines.extend(f"  {class_id}: {name}" for class_id, name in STATUS_NAMES.items())
    (output / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    label_files = sorted((source / "data").glob("room*/room*/labels.jsonl"))
    if len(label_files) != 37:
        raise RuntimeError(f"Expected 37 labels.jsonl files, found {len(label_files)}")

    counts: dict[str, Counter] = defaultdict(Counter)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    scene_counts: dict[str, Counter] = defaultdict(Counter)
    image_lists: dict[str, list[str]] = defaultdict(list)
    skip_counts = Counter()

    for labels_jsonl in label_files:
        session_dir = labels_jsonl.parent
        with labels_jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                flags = record.get("flags") or {}
                if not flags.get("labeled", False):
                    skip_counts["frame_unlabeled"] += 1
                    continue
                if flags.get("discarded", False):
                    skip_counts["frame_discarded"] += 1
                    continue
                if flags.get("high_noise", False):
                    skip_counts["frame_high_noise"] += 1
                    continue

                scene_id = record["scene_id"]
                session_id = record["session_id"]
                split = split_for_scene(scene_id)
                ir_source = session_dir / record["ir"]
                if not args.lists_only:
                    ir_source = ir_source.resolve()
                if not args.lists_only and not ir_source.is_file():
                    raise FileNotFoundError(
                        f"Missing IR image at {labels_jsonl}:{line_number}: {ir_source}"
                    )

                yolo_labels = []
                object_skips = Counter()
                has_unknown_state = False
                for obj in record.get("objects") or []:
                    yolo_label, reason = yolo_line(obj)
                    if yolo_label is None:
                        if reason == "state_unknown":
                            has_unknown_state = True
                        else:
                            object_skips[reason] += 1
                        continue
                    yolo_labels.append(yolo_label)
                    class_id = int(yolo_label.split(" ", 1)[0])
                    class_counts[split][STATUS_NAMES[class_id]] += 1

                # A visible person with no state label must not silently become
                # background supervision, so exclude its entire frame.
                if has_unknown_state:
                    skip_counts["frame_unknown_state"] += 1
                    for yolo_label in yolo_labels:
                        class_id = int(yolo_label.split(" ", 1)[0])
                        class_counts[split][STATUS_NAMES[class_id]] -= 1
                    continue
                skip_counts.update(object_skips)

                relative = Path(scene_id) / session_id / ir_source.name
                image_out = output / "images" / split / relative
                label_out = output / "labels" / split / relative.with_suffix(".txt")
                if not args.lists_only:
                    link_or_copy(ir_source, image_out, args.copy_images)
                    label_out.parent.mkdir(parents=True, exist_ok=True)
                    label_out.write_text(
                        "\n".join(yolo_labels) + ("\n" if yolo_labels else ""),
                        encoding="utf-8",
                    )
                counts[split]["images"] += 1
                counts[split]["boxes"] += len(yolo_labels)
                counts[split]["background_images"] += int(not yolo_labels)
                scene_counts[split][scene_id] += 1
                # Keep the YOLO-side symlink path. Resolving it here would
                # point back to data/.../ir and break Ultralytics' standard
                # /images/ -> /labels/ label-path mapping.
                image_lists[split].append(str(image_out))

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / f"{split}.txt").write_text(
            "\n".join(image_lists[split]) + "\n", encoding="utf-8"
        )
    write_yaml(output)
    manifest = {
        "source": str(source),
        "image_modality": "infrared pseudo-color PNG only; RGB excluded",
        "class_names": STATUS_NAMES,
        "class_names_zh": STATUS_NAMES_ZH,
        "splits": {split: dict(sorted(values.items())) for split, values in scene_counts.items()},
        "counts": {split: dict(values) for split, values in counts.items()},
        "class_counts": {split: dict(values) for split, values in class_counts.items()},
        "skipped": dict(skip_counts),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
