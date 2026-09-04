#!/usr/bin/env python3
"""Build the room04-room12 paired RGB -> thermal/pseudo-color manifest.

Unlike IR2RGB, here we keep the raw thermal ``.bin`` path as the primary
ground-truth target.  Each ``.bin`` is a ``uint16`` 62x80 temperature field,
expressed in deci-degrees Celsius and prefixed with a 4-byte header:

    celsius = uint16 / 10, skip the first 4 bytes

The pseudo-color ``ir/*.png`` is also retained so the same split can be used
for the thermal-image-synthesis baseline.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "data"
EXPECTED_SHAPE = (62, 80)
HEADER_BYTES = 4


def read_thermal(path: Path) -> np.ndarray:
    """Read a raw thermal bin into a float32 Celsius array of shape (62, 80)."""
    raw = path.read_bytes()
    expected = HEADER_BYTES + EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1] * 2
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(raw)}")
    values = np.frombuffer(raw[HEADER_BYTES:], dtype=np.uint16).astype(np.float32)
    return values.reshape(EXPECTED_SHAPE) / 10.0


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
                thermal_path = session_dir / obj["thermal"]
                rgb_path = session_dir / rgb["path"]
                if ir_path.exists() and thermal_path.exists() and rgb_path.exists():
                    pairs.append(
                        {
                            "rgb_path": str(rgb_path.resolve()),
                            "thermal_path": str(thermal_path.resolve()),
                            "ir_path": str(ir_path.resolve()),
                            "frame_id": obj["frame_id"],
                            "scene_id": obj["scene_id"],
                            "session_id": obj["session_id"],
                        }
                    )
    return pairs


def compute_thermal_stats(rows: list[dict], out: Path) -> None:
    """Per-pixel mean/std and global min/max over the train split only."""
    count = 0
    mean = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    m2 = np.zeros(EXPECTED_SHAPE, dtype=np.float64)
    global_min = float("inf")
    global_max = float("-inf")

    for row in tqdm(rows, desc="thermal stats"):
        field = read_thermal(Path(row["thermal_path"])).astype(np.float64)
        count += 1
        delta = field - mean
        mean += delta / count
        delta2 = field - mean
        m2 += delta * delta2
        global_min = min(global_min, float(field.min()))
        global_max = max(global_max, float(field.max()))

    std = np.sqrt(m2 / count)
    std = np.clip(std, 1e-6, None)
    np.savez_compressed(
        out,
        pixel_mean=mean.astype(np.float32),
        pixel_std=std.astype(np.float32),
        global_min=np.float32(global_min),
        global_max=np.float32(global_max),
        count=np.int64(count),
    )
    print(f"thermal stats -> {out}")
    print(
        f"global temperature range: {global_min:.2f} .. {global_max:.2f} C; "
        f"per-pixel std range: {float(std.min()):.3f} .. {float(std.max()):.3f} C"
    )


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
                fieldnames=[
                    "rgb_path",
                    "thermal_path",
                    "ir_path",
                    "frame_id",
                    "scene_id",
                    "session_id",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} pairs -> {path}")

    compute_thermal_stats(train, OUT / "thermal_stats.npz")


if __name__ == "__main__":
    main()
