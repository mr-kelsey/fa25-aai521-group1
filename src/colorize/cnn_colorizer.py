# src/colorize/cnn_colorizer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


class CNNColorizerNet(nn.Module):
    """
    Simple fully-convolutional CNN that maps a 1-channel grayscale image
    to a 3-channel RGB image with outputs in [0, 1] via Sigmoid.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.activation(x)
        return x


class CNNColorizer:
    """
    Wrapper around CNNColorizerNet for PIL-based grayscale -> RGB colorization.
    """

    def __init__(self, weights_path: str | Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNColorizerNet().to(self.device)

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Colorizer weights not found at {weights_path}. "
                f"Train the model first to create this file."
            )

        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Training/inference resolution (fully-conv, but we normalize via resize)
        self.infer_size = 64

    def _prepare_input(self, img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Convert to grayscale and remember original size
        img_gray = img.convert("L")
        orig_w, orig_h = img_gray.size

        # Resize to training size
        img_resized = img_gray.resize((self.infer_size, self.infer_size), Image.BILINEAR)

        # To numpy [H, W] in [0,1]
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        # To tensor [1,1,H,W]
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor, (orig_w, orig_h)

    def _to_pil(self, tensor: torch.Tensor, orig_size: Tuple[int, int]) -> Image.Image:
        # tensor: [1,3,H,W] in [0,1]
        tensor = tensor.detach().cpu().squeeze(0)  # [3,H,W]
        arr = tensor.numpy()
        arr = np.transpose(arr, (1, 2, 0))  # [H,W,3]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        img_small = Image.fromarray(arr, mode="RGB")
        img_color = img_small.resize(orig_size, Image.BILINEAR)
        return img_color

    def colorize(self, img: Image.Image) -> Image.Image:
        """
        Colorize a PIL image (treated as grayscale regardless of mode).
        Returns a new RGB PIL image.
        """
        x, orig_size = self._prepare_input(img)

        with torch.no_grad():
            y = self.model(x)

        out_img = self._to_pil(y, orig_size)
        return out_img
