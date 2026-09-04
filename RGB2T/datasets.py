"""Paired RGB -> thermal/pseudo-color datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_H, IMAGE_W = 192, 256
THERMAL_SHAPE = (62, 80)


def read_thermal(path: Path) -> np.ndarray:
    """Read a raw thermal bin into a float32 Celsius array of shape (62, 80)."""
    raw = path.read_bytes()
    values = np.frombuffer(raw[4:], dtype=np.uint16).astype(np.float32)
    return values.reshape(THERMAL_SHAPE) / 10.0


class RGB2TDataset(Dataset):
    """Paired dataset.

    ``target`` is either:
      * ``"thermal"``: 1-channel per-pixel z-scored temperature field.
      * ``"pseudo"``: 3-channel pseudo-color thermal image in [-1, 1].
    """

    def __init__(self, csv_path: Path, target: str = "thermal", training: bool = True) -> None:
        if target not in {"thermal", "pseudo"}:
            raise ValueError(f"unsupported target: {target}")
        self.target = target
        self.training = training
        with csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))

        stats = np.load(csv_path.parent / "thermal_stats.npz")
        self.pixel_mean = stats["pixel_mean"].astype(np.float32)
        self.pixel_std = stats["pixel_std"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _resize_rgb(self, rgb: Image.Image) -> torch.Tensor:
        rgb = rgb.convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        return (transforms.functional.to_tensor(rgb) - 0.5) / 0.5

    def _thermal_target(self, path: Path) -> torch.Tensor:
        field = read_thermal(path)  # (62, 80) Celsius
        norm = (field - self.pixel_mean) / self.pixel_std
        tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)  # 1x62x80
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(IMAGE_H, IMAGE_W),
            mode="bilinear",
            align_corners=False,
        )[0]
        return tensor

    def _pseudo_target(self, path: Path) -> torch.Tensor:
        ir = Image.open(path).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        return (transforms.functional.to_tensor(ir) - 0.5) / 0.5

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = self._resize_rgb(Image.open(row["rgb_path"]))
        if self.target == "thermal":
            target = self._thermal_target(Path(row["thermal_path"]))
        else:
            target = self._pseudo_target(Path(row["ir_path"]))

        if self.training and random.random() < 0.5:
            rgb = transforms.functional.hflip(rgb)
            target = transforms.functional.hflip(target)
        return rgb, target


class RGB2TRelDataset(Dataset):
    """ThermalGAN-style relative-temperature dataset.

    The condition is ``[RGB, baseline]`` where the baseline is the per-pixel
    mean thermal field rendered as an extra channel.  The target is the
    per-pixel z-scored temperature residual, so this is a four-channel-to-one
    conditional GAN problem.
    """

    def __init__(self, csv_path: Path, training: bool = True) -> None:
        self.training = training
        with csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        stats = np.load(csv_path.parent / "thermal_stats.npz")
        self.pixel_mean = stats["pixel_mean"].astype(np.float32)
        self.pixel_std = stats["pixel_std"].astype(np.float32)
        global_min = float(stats["global_min"])
        global_max = float(stats["global_max"])
        self.baseline = (self.pixel_mean - global_min) / (global_max - global_min) * 2.0 - 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        rgb = Image.open(row["rgb_path"]).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        rgb = (transforms.functional.to_tensor(rgb) - 0.5) / 0.5

        field = read_thermal(Path(row["thermal_path"]))
        norm = (field - self.pixel_mean) / self.pixel_std
        target = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)
        target = F.interpolate(target.unsqueeze(0), size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)[0]

        baseline = torch.from_numpy(self.baseline).unsqueeze(0)
        baseline = F.interpolate(baseline.unsqueeze(0), size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)[0]
        condition = torch.cat([rgb, baseline], dim=0)

        if self.training and random.random() < 0.5:
            condition = transforms.functional.hflip(condition)
            target = transforms.functional.hflip(target)
        return condition, target


class RGB2TInferenceDataset(RGB2TDataset):
    """Like RGB2TDataset but additionally returns the source row for per-scene metrics."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rgb, target = super().__getitem__(index)
        return rgb, target, self.rows[index]


class RGB2TRelInferenceDataset(RGB2TRelDataset):
    """Like RGB2TRelDataset but additionally returns the source row for per-scene metrics."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        condition, target = super().__getitem__(index)
        return condition, target, self.rows[index]
