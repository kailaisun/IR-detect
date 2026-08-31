# Transparent non-converged model snapshots

This directory reports real checkpoints captured before convergence. These
results are supplementary training-progress references and **do not replace**
the best-checkpoint benchmark results elsewhere in the repository.

All models use only 80×62 three-channel infrared pseudo-color images. Every
number is measured on the same untouched scene-disjoint test rooms 03, 10, and
18: 12,237 images and 16,657 person instances. RGB is never model input.

## Metric protocol

- `mAP50` and `mAP50–95` use standard COCO box evaluation.
- Precision, Recall, and F1 use IoU 0.5 at one confidence threshold selected to
  maximize macro class F1.
- Per-class Precision and Recall use that same global confidence threshold.
- Latency uses one NVIDIA L40S, strict batch size 1, 100 warm-up frames, and
  1,000 preloaded infrared frames repeated three times. It includes the
  framework ndarray pipeline, forward pass, and postprocessing, but excludes
  disk I/O.
- The full-precision training checkpoint is evaluated first; the compact
  published checkpoint is then loaded from its final repository path and
  evaluated again. The JSON files below come from the published weights.

## Selected-stage overview

The stages are intentionally different. This table was assembled to inspect
representative non-converged behavior around requested mAP50 levels; it is not
a compute-matched architecture ranking.

| Method | Stage | Precision | Recall | Macro F1 | mAP50 | mAP50–95 | Latency | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s | epoch 11 | 0.704 | 0.681 | 0.692 | 0.705 | 0.427 | **5.339 ms** | **187.3** |
| YOLO26s | epoch 11 | 0.778 | 0.680 | 0.723 | 0.751 | 0.480 | 5.828 ms | 171.6 |
| RT-DETR-L | epoch 4 | 0.733 | 0.633 | 0.668 | 0.692 | 0.393 | 11.659 ms | 85.8 |
| RTMDet-s | epoch 14 | 0.751 | 0.666 | 0.699 | 0.742 | 0.452 | 12.049 ms | 83.0 |
| DINO 4-scale R50 | iteration 200 | 0.686 | 0.613 | 0.630 | 0.681 | 0.438 | 24.378 ms | 41.0 |
| Faster R-CNN R50-FPN | epoch 1 | 0.751 | 0.649 | 0.640 | 0.723 | 0.412 | 13.036 ms | 76.7 |

The recovered target-region snapshots are therefore:

- YOLO11s: mAP50 0.705;
- YOLO26s: mAP50 0.751;
- RT-DETR-L: mAP50 0.692;
- RTMDet-s: mAP50 0.742;
- DINO: mAP50 0.681;
- Faster R-CNN: mAP50 0.723.

## Complete per-class metrics

| Method | Class | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11s | `lie` | 0.867 | 0.782 | 0.822 | 0.852 | 0.509 |
| YOLO11s | `sit` | 0.754 | 0.790 | 0.772 | 0.815 | 0.510 |
| YOLO11s | `other` | 0.498 | 0.468 | 0.482 | 0.452 | 0.294 |
| YOLO11s | `off_bed` | 0.698 | 0.685 | 0.691 | 0.702 | 0.398 |
| YOLO26s | `lie` | 0.826 | 0.802 | 0.814 | 0.856 | 0.530 |
| YOLO26s | `sit` | 0.803 | 0.755 | 0.779 | 0.840 | 0.563 |
| YOLO26s | `other` | 0.634 | 0.549 | 0.589 | 0.562 | 0.387 |
| YOLO26s | `off_bed` | 0.847 | 0.615 | 0.713 | 0.748 | 0.440 |
| RT-DETR-L | `lie` | 0.884 | 0.680 | 0.769 | 0.795 | 0.449 |
| RT-DETR-L | `sit` | 0.748 | 0.719 | 0.733 | 0.781 | 0.451 |
| RT-DETR-L | `other` | 0.690 | 0.395 | 0.502 | 0.511 | 0.341 |
| RT-DETR-L | `off_bed` | 0.611 | 0.738 | 0.669 | 0.680 | 0.330 |
| RTMDet-s | `lie` | 0.872 | 0.715 | 0.786 | 0.860 | 0.487 |
| RTMDet-s | `sit` | 0.721 | 0.758 | 0.739 | 0.794 | 0.518 |
| RTMDet-s | `other` | 0.715 | 0.457 | 0.558 | 0.544 | 0.360 |
| RTMDet-s | `off_bed` | 0.693 | 0.734 | 0.713 | 0.771 | 0.442 |
| DINO 4-scale R50 | `lie` | 0.673 | 0.846 | 0.750 | 0.840 | 0.533 |
| DINO 4-scale R50 | `sit` | 0.678 | 0.752 | 0.713 | 0.764 | 0.531 |
| DINO 4-scale R50 | `other` | 0.547 | 0.296 | 0.384 | 0.379 | 0.273 |
| DINO 4-scale R50 | `off_bed` | 0.848 | 0.558 | 0.673 | 0.739 | 0.412 |
| Faster R-CNN R50-FPN | `lie` | 0.676 | 0.905 | 0.774 | 0.897 | 0.482 |
| Faster R-CNN R50-FPN | `sit` | 0.714 | 0.839 | 0.771 | 0.834 | 0.486 |
| Faster R-CNN R50-FPN | `other` | 0.828 | 0.174 | 0.288 | 0.387 | 0.264 |
| Faster R-CNN R50-FPN | `off_bed` | 0.786 | 0.679 | 0.728 | 0.773 | 0.415 |

## Matched epoch-1 reference

For a genuinely equal-stage comparison, the original epoch-1 snapshots are
retained separately:

| Method | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| YOLO26s | 0.651 | 0.587 | 0.644 | 0.399 |
| RT-DETR-L | 0.548 | 0.538 | 0.547 | 0.314 |
| YOLO11s | 0.557 | 0.510 | 0.508 | 0.258 |

## Checkpoint recovery

YOLO26s already retained its epoch-11 checkpoint. RT-DETR-L retained only
epochs 1, 11, and 21, so epoch 4 was reproduced with the same 100-epoch
schedule and `save_period=1`. MMDetection had been configured with
`max_keep_ckpts=2`, so its requested early checkpoints were also reproduced
from the same official COCO initialization, data split, seed, optimizer,
augmentation, batch size, and learning-rate schedule, changing only checkpoint
retention.

Published recovery configurations:

- `retrain_configs/rtmdet_s_save_every_epoch.py`;
- `retrain_configs/faster_rcnn_save_every_epoch.py`;
- `retrain_configs/dino_save_every_epoch.py`;
- `retrain_configs/dino_save_every_100_iters.py`.

## Artifacts

Each selected snapshot directory contains its compact inference checkpoint,
complete machine-readable metrics, and `benchmark.json` latency results:

- `yolo11s_epoch11/`;
- `yolo26s_epoch11/`;
- `rtdetr_l_epoch4/`;
- `rtmdet_s_epoch14/`;
- `dino_iter200/`;
- `faster_rcnn_epoch1/`.

Additional artifacts:

- `summary.json`: normalized overall and per-class metrics for all snapshots;
- `selected_metrics.csv`: flat overall and per-class metrics for analysis;
- `table_selected_stage.tex`: paper table with overall and class-wise mAP50;
- `table_selected_per_class.tex`: complete per-class P/R/F1/mAP table;
- `table_epoch1.tex`: fair matched-epoch reference table.

The earlier epoch-1 weights, PR curves, confusion matrices, and prediction
samples remain available in their original directories. All best-checkpoint
results remain unchanged.
