# Transparent early-training snapshots

These checkpoints show how the three Ultralytics detectors behave before
convergence. They are supplementary learning-curve artifacts and **do not
replace** the best-checkpoint results in the main benchmark READMEs.

All numbers are measured on the same untouched, scene-disjoint test rooms 03,
10, and 18: 12,237 infrared images and 16,657 annotated person instances. RGB
is not used by any model.

## Matched epoch-1 comparison

This is the fair early-training comparison: every model is evaluated after its
first completed training epoch.

| Method | Epoch | Precision | Recall | mAP50 | mAP50–95 | Lie mAP50 | Sit mAP50 | Other mAP50 | Off-bed mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26s | 1 | **0.651** | **0.587** | **0.644** | **0.399** | **0.813** | **0.711** | **0.343** | **0.708** |
| RT-DETR-L | 1 | 0.548 | 0.538 | 0.547 | 0.314 | 0.806 | 0.664 | 0.083 | 0.636 |
| YOLO11s | 1 | 0.557 | 0.510 | 0.508 | 0.258 | 0.666 | 0.462 | 0.256 | 0.648 |

At equal one-epoch training budget, YOLO26s converges fastest. RT-DETR-L has
almost no usable `other` recall at this point, while YOLO11s still requires
additional optimization before reaching its stable ranking.

## YOLO11 near-0.70 preview

The available checkpoints are saved every ten epochs. The YOLO11s
`epoch10.pt` file represents the state after epoch 11 and is the closest saved
checkpoint to an overall test mAP50 of 0.70:

| Method | Epoch | Precision | Recall | mAP50 | mAP50–95 | Lie mAP50 | Sit mAP50 | Other mAP50 | Off-bed mAP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s | 11 | **0.704** | **0.681** | **0.705** | **0.427** | **0.852** | **0.815** | **0.452** | 0.702 |
| YOLO26s | 1 | 0.651 | 0.587 | 0.644 | 0.399 | 0.813 | 0.711 | 0.343 | **0.708** |
| RT-DETR-L | 1 | 0.548 | 0.538 | 0.547 | 0.314 | 0.806 | 0.664 | 0.083 | 0.636 |

This second table intentionally mixes training stages and is therefore only a
visual preview of early model behavior, **not a fair architecture ranking**.
The epoch column must be retained whenever these values are quoted.

Compared with the final best checkpoints:

| Method | Early epoch | Early mAP50 | Final mAP50 | Early mAP50–95 | Final mAP50–95 |
|---|---:|---:|---:|---:|---:|
| YOLO11s | 11 | 0.705 | 0.789 | 0.427 | 0.505 |
| YOLO26s | 1 | 0.644 | 0.784 | 0.399 | 0.507 |
| RT-DETR-L | 1 | 0.547 | 0.775 | 0.314 | 0.499 |

## Artifacts

- `yolo26s_epoch1/`: epoch-1 YOLO26s weight, metrics, PR curve, confusion
  matrix, and test prediction sample.
- `yolo11s_epoch1/`: matched epoch-1 YOLO11s snapshot.
- `yolo11s_epoch11/`: YOLO11s snapshot with test mAP50 approximately 0.70.
- `rtdetr_l_epoch1/`: epoch-1 RT-DETR-L snapshot.
- `summary.json`: complete machine-readable metrics and artifact paths.
- `table_epoch1.tex`: fair matched-epoch LaTeX table.
- `table_selected_stage.tex`: explicitly unequal-stage preview table.

The published `.pt` files are inference-only FP16-storage checkpoints. Each was
loaded again from its final repository path and re-evaluated to produce the
included `metrics.json`.

