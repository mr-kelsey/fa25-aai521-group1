"""Utilities for loading and running a Stable Diffusion img2img pipeline for denoising."""

from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def load_denoise_model(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionImg2ImgPipeline:
    """Load Stable Diffusion img2img pipeline for denoising with appropriate device/dtype."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def run_denoising(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    prompt: str = "high quality clean photo, detailed, sharp",
    strength: float = 0.4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
):
    """Run denoising given a PIL image and return the denoised PIL image."""
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    return result.images[0]
