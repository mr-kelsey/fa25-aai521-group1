"""BM3D-based image denoiser."""

from typing import Optional

from PIL import Image
import numpy as np
from bm3d import bm3d


def load_denoise_model(model_id: Optional[str] = None):
    return "bm3d"


def run_denoising(model, image: Image.Image) -> Image.Image:
    img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    denoised = bm3d(img, sigma_psd=0.20)
    out = (denoised * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(out)
