from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


CLASS_NAMES = ("one_person", "two_people")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(image_size: int, training: bool) -> transforms.Compose:
    resized_height = round(image_size * 62 / 80)
    pad_total = image_size - resized_height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    operations: list = [
        transforms.Resize((resized_height, image_size), antialias=True),
        transforms.Pad((0, pad_top, 0, pad_bottom), fill=(114, 114, 114)),
    ]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    fill=(114, 114, 114),
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return transforms.Compose(operations)


class PersonCountDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        dataset_root: Path,
        transform: transforms.Compose,
        cache: bool = False,
    ) -> None:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.samples = [(dataset_root / row["image"], int(row["label"])) for row in rows]
        self.transform = transform
        self.cached_images = None
        if cache:
            cached_images = []
            for index, (path, _) in enumerate(self.samples, 1):
                with Image.open(path) as image:
                    cached_images.append(np.asarray(image.convert("RGB")).copy())
                if index % 10000 == 0 or index == len(self.samples):
                    print(f"cached {index}/{len(self.samples)} images from {csv_path.name}")
            self.cached_images = cached_images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        if self.cached_images is None:
            with Image.open(path) as image:
                image_rgb = image.convert("RGB")
                tensor = self.transform(image_rgb)
        else:
            tensor = self.transform(Image.fromarray(self.cached_images[index]))
        return tensor, label


def build_loader(
    csv_path: Path,
    dataset_root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    training: bool,
    cache: bool,
) -> DataLoader:
    dataset = PersonCountDataset(
        csv_path,
        dataset_root,
        build_transform(image_size, training),
        cache=cache,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=training,
    )


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[positives].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[label, prediction] += 1
    per_class = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[class_id, :].sum()),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
    accuracy = float(np.trace(confusion) / confusion.sum())
    count_error = predictions.astype(np.float64) - labels
    return {
        "accuracy": accuracy,
        "exact_count_accuracy": accuracy,
        "mae": float(np.abs(count_error).mean()),
        "mse": float(np.square(count_error).mean()),
        "rmse": float(np.sqrt(np.square(count_error).mean())),
        "mean_error": float(count_error.mean()),
        "balanced_accuracy": float(np.mean([value["recall"] for value in per_class.values()])),
        "macro_precision": float(np.mean([value["precision"] for value in per_class.values()])),
        "macro_recall": float(np.mean([value["recall"] for value in per_class.values()])),
        "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
        "roc_auc_two_people": binary_auc(labels, probabilities[:, 1]),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    predictions_all = []
    probabilities_all = []
    loss_sum = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels) if criterion is not None else None
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        if loss is not None:
            loss_sum += float(loss) * len(labels)
        sample_count += len(labels)
        labels_all.append(labels.cpu().numpy())
        predictions_all.append(predictions.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    probabilities_np = np.concatenate(probabilities_all)
    metrics = classification_metrics(labels_np, predictions_np, probabilities_np)
    if criterion is not None:
        metrics["loss"] = loss_sum / sample_count
    return metrics, labels_np, predictions_np, probabilities_np
