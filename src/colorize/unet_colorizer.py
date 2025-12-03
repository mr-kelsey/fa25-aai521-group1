# src/colorize/unet_colorizer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ---------- UNet backbone ----------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetColorizerNet(nn.Module):
    """
    UNet-like architecture for 1-channel -> 3-channel colorization.
    Input:  [B,1,H,W] grayscale
    Output: [B,3,H,W] RGB in [0,1] via Sigmoid
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)

        out = self.out_conv(u1)
        out = self.activation(out)
        return out


# ---------- Wrapper for use in pipeline ----------

class UNetColorizer:
    """
    High-level wrapper that:
      - loads trained UNetColorizerNet weights
      - takes PIL grayscale image
      - returns PIL RGB image
    """

    def __init__(self, weights_path: str | Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetColorizerNet().to(self.device)

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"UNet colorizer weights not found: {weights_path}\n"
                f"Run train_unet_colorizer.py first to create this file."
            )

        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.input_size = 256  # training resolution

    def _prepare_input(self, img: Image.Image):
        grey = img.convert("L")
        orig_w, orig_h = grey.size

        grey_resized = grey.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(grey_resized, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        tensor = tensor.to(self.device)
        return tensor, (orig_w, orig_h)

    def _to_pil(self, tensor: torch.Tensor, orig_size: Tuple[int, int]) -> Image.Image:
        tensor = tensor.detach().cpu().squeeze(0)  # [3,H,W]
        arr = tensor.numpy()
        arr = np.transpose(arr, (1, 2, 0))  # [H,W,3]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        img_small = Image.fromarray(arr, mode="RGB")
        img_color = img_small.resize(orig_size, Image.BILINEAR)
        return img_color

    def colorize(self, img: Image.Image) -> Image.Image:
        x, orig_size = self._prepare_input(img)
        with torch.no_grad():
            y = self.model(x)
        return self._to_pil(y, orig_size)
