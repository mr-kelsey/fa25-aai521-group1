"""Single-image denoising pipeline.

Loads a noisy image, runs a Stable Diffusion img2img denoising model,
optionally saves and visualizes the result.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .model_utils import load_denoise_model, run_denoising
from src.inpaint.viz_utils import show_before_after


def _load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def denoise_image(
    image_path: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    save_to: Optional[str] = None,
    visualize: bool = False,
    prompt: str = "high quality clean photo, detailed, sharp",
    strength: float = 0.4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> Tuple[Image.Image, Image.Image]:
    """Denoise a single image using a Stable Diffusion img2img pipeline.

    Args:
        image_path: Path to noisy input image.
        model_id: Hugging Face model id for the img2img pipeline.
        save_to: Base path to save outputs.
        visualize: Show before/after comparison.
        prompt: Optional text prompt to guide denoising.
        strength: Img2img strength parameter.
        guidance_scale: Guidance scale for classifier-free guidance.
        num_inference_steps: Number of diffusion steps.

    Returns:
        (noisy_image_pil, denoised_image_pil)
    """
    noisy_pil = _load_image_pil(image_path)

    pipe = load_denoise_model(model_id=model_id)
    denoised_pil = run_denoising(
        pipe,
        image=noisy_pil,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        noisy_pil.save(base.with_name(base.stem + "_input.png"))
        denoised_pil.save(base.with_name(base.stem + "_denoised.png"))

    if visualize:
        show_before_after(noisy_pil, denoised_pil, title_left="Noisy", title_right="Denoised")

    return noisy_pil, denoised_pil
