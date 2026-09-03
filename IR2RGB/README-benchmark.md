# IR2RGB — baseline benchmark

Generating the synchronized visible RGB image from the 80x62 thermal-infrared
pseudo-color image. All models are trained on the same room04-room12 pairs
(train 30,019 / test 7,505, random 80/20 split).

## Results

| Model | Family | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Eval set |
|---|---|---:|---:|---:|---|
| Pix2Pix | GAN (regression) | 18.59 | 0.7001 | 0.1828 | 7,505 |
| NAFNet (epoch 28) | CNN restoration | 15.80 | 0.4267 | 0.5512 | 1,000 |
| SDXL ControlNet | pretrained diffusion | 10.62 | 0.4163 | 0.6893 | 1,000 |
| ControlNet SD1.5 | pretrained diffusion | 10.54 | 0.3670 | 0.7027 | 1,000 |
| Palette (simple DDPM) | diffusion from scratch | 7.85 | 0.1397 | 0.7635 | 500 |

Pix2Pix full-suite metrics are in `results/metrics.json` (PSNR 18.59,
SSIM 0.7001, MS-SSIM 0.7847, MAE 0.0617, RMSE 0.1179, LPIPS 0.1828,
Delta E 6.7026, FID 28.8552, KID 0.0159).

NAFNet is still training (100 epochs); the number above is from its current
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
