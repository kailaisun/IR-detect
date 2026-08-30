#!/usr/bin/env python3
"""Convert the existing scene-disjoint IR-only YOLO dataset to COCO JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_NAMES = ("lie", "sit", "other", "off_bed")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=root / "dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "mmdetection_data",
    )
    return parser.parse_args()


def convert_split(yolo_root: Path, output: Path, split: str) -> dict:
    image_root = yolo_root / "images" / split
    label_root = yolo_root / "labels" / split
    image_paths = [Path(line) for line in (yolo_root / f"{split}.txt").read_text().splitlines()]
    images: list[dict] = []
    annotations: list[dict] = []
    class_counts = Counter()
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, 1):
        try:
            relative = image_path.relative_to(image_root)
        except ValueError as exc:
            raise RuntimeError(f"Image is outside {image_root}: {image_path}") from exc
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": relative.as_posix(),
                "width": width,
                "height": height,
            }
        )

        label_path = label_root / relative.with_suffix(".txt")
        for line in label_path.read_text().splitlines():
            class_id_text, xc_text, yc_text, bw_text, bh_text = line.split()
            class_id = int(class_id_text)
            xc, yc, bw, bh = map(float, (xc_text, yc_text, bw_text, bh_text))
            box_width = bw * width
            box_height = bh * height
            x = max(0.0, xc * width - box_width / 2)
            y = max(0.0, yc * height - box_height / 2)
            box_width = min(box_width, width - x)
            box_height = min(box_height, height - y)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            class_counts[CLASS_NAMES[class_id]] += 1

    coco = {
        "info": {
            "description": "household_ir person state detection; IR-only input",
            "image_modality": "infrared pseudo-color PNG only; RGB excluded",
        },
        "licenses": [],
        "categories": [
            {"id": class_id + 1, "name": name, "supercategory": "person_state"}
            for class_id, name in enumerate(CLASS_NAMES)
        ],
        "images": images,
        "annotations": annotations,
    }
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / f"instances_{split}.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "images": len(images),
        "boxes": len(annotations),
        "class_counts": dict(class_counts),
    }


def main() -> None:
    args = parse_args()
    yolo_root = args.yolo_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    image_link = output / "images"
    target = yolo_root / "images"
    if image_link.is_symlink():
        if image_link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected image symlink target: {image_link.resolve()}")
    elif image_link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {image_link}")
    else:
        image_link.symlink_to(target)

    manifest = {
        "source": str(yolo_root),
        "image_modality": "infrared pseudo-color PNG only; RGB excluded",
        "classes": list(CLASS_NAMES),
        "splits": {
            split: convert_split(yolo_root, output, split)
            for split in ("train", "val", "test")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
