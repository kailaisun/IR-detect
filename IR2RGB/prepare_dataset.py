#!/usr/bin/env python3
"""Build the room04-room12 paired IR-pseudo-color -> RGB manifest and split."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "data"


def collect_pairs() -> list[dict]:
    pairs: list[dict] = []
    for room_id in range(4, 13):
        scene = f"room{room_id:02d}"
        scene_dir = DATA / scene
        for session_dir in sorted(scene_dir.iterdir()):
            labels = session_dir / "labels.jsonl"
            if not session_dir.is_dir() or not labels.exists():
                continue
            for line in labels.open():
                obj = json.loads(line)
                rgb = obj.get("rgb")
                if not rgb or rgb.get("kind") != "device_cam" or not rgb.get("ok", True):
                    continue
                ir_path = session_dir / obj["ir"]
                rgb_path = session_dir / rgb["path"]
                if ir_path.exists() and rgb_path.exists():
                    pairs.append(
                        {
                            "ir_path": str(ir_path.resolve()),
                            "rgb_path": str(rgb_path.resolve()),
                            "frame_id": obj["frame_id"],
                            "scene_id": obj["scene_id"],
                            "session_id": obj["session_id"],
                        }
                    )
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs()
    random.Random(42).shuffle(pairs)
    split = int(0.8 * len(pairs))
    train, test = pairs[:split], pairs[split:]
    for name, rows in (("train", train), ("test", test)):
        path = OUT / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["ir_path", "rgb_path", "frame_id", "scene_id", "session_id"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} pairs -> {path}")


if __name__ == "__main__":
    main()
