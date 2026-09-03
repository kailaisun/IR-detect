# IR2RGB — baseline benchmark

Generating the synchronized visible RGB image from the 80x62 thermal-infrared
pseudo-color image. All models are trained on the same room04-room12 pairs
(train 30,019 / test 7,505, random 80/20 split).

## Results

All metrics are on the held-out test split. Pix2Pix uses all 7,505 test
images; the other four models use the same 1,000-image subset.

| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | MAE ↓ | RMSE ↓ | LPIPS ↓ | ΔE ↓ | FID ↓ | KID ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pix2Pix | 18.59 | 0.7001 | 0.7847 | 0.0617 | 0.1179 | 0.1828 | 6.70 | 28.86 | 0.0159 |
| NAFNet (epoch 28) | 15.80 | 0.4256 | 0.4997 | 0.1145 | 0.1677 | 0.5512 | 11.37 | 270.35 | 0.2886 |
| SDXL ControlNet | 10.52 | 0.3983 | 0.1855 | 0.2374 | 0.3016 | 0.7011 | 22.83 | 68.79 | 0.0383 |
| ControlNet SD1.5 | 10.22 | 0.3334 | 0.2066 | 0.2517 | 0.3237 | 0.7375 | 24.22 | 78.28 | 0.0384 |
| Palette (simple DDPM) | 7.83 | 0.1371 | 0.1291 | 0.3374 | 0.4225 | 0.7609 | 36.21 | 229.05 | 0.2037 |

Machine-readable full metrics are in `results/summary_full.json` and
`results/<model>/metrics.json`.

NAFNet is still training (100 epochs); its number above is from the current
best checkpoint at epoch 28 and will be updated when training finishes.

### Important note on diffusion models

ControlNet and Palette use an empty text prompt and generate a plausible RGB
that follows the infrared structure, not a pixel-aligned reconstruction of the
ground truth. PSNR / SSIM are therefore naturally much lower than for
regression models such as Pix2Pix. For these models, FID / distribution and
qualitative samples are the meaningful comparisons.

### Known limitation

The input is only 80x62 pixels, so a person occupies a handful of pixels and a
face is a 2-4 pixel blob. Facial details cannot be recovered and generated
faces are therefore distorted or blurry; this is an inherent limitation of the
input resolution, not a model bug.

## Visualizations

Each image is a row of `IR input | generated RGB | real RGB`.

### Pix2Pix

![Pix2Pix examples](examples.png)

### NAFNet

![NAFNet examples](results/nafnet/samples.png)

### SDXL ControlNet

![SDXL ControlNet examples](results/controlnet_sdxl/samples.png)

### ControlNet SD1.5

![ControlNet SD1.5 examples](results/controlnet_sd15/samples.png)

### Palette (simple DDPM)

![Palette examples](results/palette/samples.png)

## Files

- `train.py` / `pix2pix.py` / `evaluate.py`: Pix2Pix pipeline.
- `train_controlnet_local.py` / `evaluate_controlnet.py`: SD1.5 ControlNet.
- `train_controlnet_sdxl_local.py` / `evaluate_controlnet_sdxl.py`: SDXL ControlNet.
- `train_palette.py` / `evaluate_palette.py`: Palette-style conditional DDPM.
- `train_nafnet.py`: NAFNet-style restoration network.
- `results/`: per-model metrics and sample images.
