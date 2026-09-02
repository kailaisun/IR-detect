"""Pix2Pix generator / discriminator and the paired image dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = (192, 256)  # (height, width), aspect-preserving for 640x480 RGB


class UNetGenerator(nn.Module):
    """U-Net generator with skip connections, as in the original Pix2Pix."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_downs: int = 6,
        ngf: int = 64,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            out_channels, ngf, input_nc=in_channels, submodule=unet_block,
            outermost=True, norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field by default)."""

    def __init__(self, in_channels: int = 6, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(model: nn.Module, gain: float = 0.02) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, gain)
            nn.init.constant_(module.bias, 0.0)


class PairedDataset(Dataset):
    def __init__(self, csv_path: Path, training: bool = True) -> None:
        with csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        ir = Image.open(row["ir_path"]).convert("RGB")
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        h, w = IMAGE_SIZE
        ir = ir.resize((w, h), Image.BILINEAR)
        rgb = rgb.resize((w, h), Image.BILINEAR)
        if self.training and random.random() < 0.5:
            ir = transforms.functional.hflip(ir)
            rgb = transforms.functional.hflip(rgb)
        ir_t = (transforms.functional.to_tensor(ir) - 0.5) / 0.5
        rgb_t = (transforms.functional.to_tensor(rgb) - 0.5) / 0.5
        return ir_t, rgb_t
