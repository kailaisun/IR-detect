# IR2RGB: thermal infrared -> visible RGB translation

Generate the synchronized visible RGB image from the low-resolution thermal
infrared pseudo-color image. This is a paired image-to-image translation task
built on the `household_ir` dataset.

## Task

- Input: 80x62 thermal infrared pseudo-color PNG (three channels).
- Output: 640x480 visible RGB image from the on-device camera.
- The two sensors are on the same device and are pixel-aligned; only the
  `device_cam` pairs are used.
- The mapping is ill-posed: many plausible RGB colors can explain one thermal
  frame, so perceptual / distributional metrics matter as much as PSNR.

## Data

- Scenes: `room04` ... `room12` (18 sessions).
- Frames: every frame whose `rgb.kind == device_cam` and `rgb.ok == true`.
  Total **37,524** pairs.
- Split: random 80 / 20 at frame level (seed 42):
  - train: **30,019**
  - test: **7,505**
- Manifest: `data/train.csv`, `data/test.csv` (built by `prepare_dataset.py`).
- Training resolution: both images are resized to **256x192** (4:3).

## Method

Pix2Pix baseline:

- Generator: U-Net with skip connections, 6 downsampling levels.
- Discriminator: PatchGAN (70x70 receptive field).
- Loss: adversarial (BCE) + L1 x 100.
- Optimizer: Adam, lr 2e-4, betas (0.5, 0.999), linear lr decay after epoch 50.
- Batch size 32, 100 epochs on one NVIDIA H200.

## Results (held-out test, 7,505 images)

| Metric | Value |
|---|---:|
| PSNR (dB) ↑ | 18.59 |
| SSIM ↑ | 0.7001 |
| LPIPS ↓ | 0.1828 |
| FID ↓ | 28.91 |

## Visualizations

![examples](examples.png)

The full-metric JSON is [`results/metrics.json`](results/metrics.json). A
larger test grid is [`results/test_samples.png`](results/test_samples.png).
Per-epoch training samples and checkpoints are kept locally under `runs/`.

## Reproduction

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision pillow numpy matplotlib tqdm scipy lpips scikit-image

.venv/bin/python prepare_dataset.py
.venv/bin/python train.py --device cuda:4 --batch-size 32 --epochs 100 --workers 8
.venv/bin/python evaluate.py --device cuda:4 --batch-size 64 --workers 8
```

## Files

- `prepare_dataset.py`: build the paired manifest and random 80/20 split.
- `pix2pix.py`: U-Net generator, PatchGAN discriminator, paired dataset.
- `train.py`: Pix2Pix training loop.
- `evaluate.py`: PSNR / SSIM / LPIPS / FID evaluation and sample generation.
