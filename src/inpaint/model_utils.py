"""
model_utils.py

Utilities for loading and running Stable Diffusion Inpainting.
"""

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def load_inpaint_model(model_id: str = "stabilityai/stable-diffusion-inpainting") -> StableDiffusionInpaintPipeline:
    """
    Load Stable Diffusion Inpainting pipeline with appropriate device/dtype.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    return pipe.to(device)


def run_inpainting(pipe: StableDiffusionInpaintPipeline, image: Image.Image, mask: Image.Image, prompt: str = "") -> Image.Image:
    """
    Run inpainting given a PIL image and corresponding PIL mask.

    Args:
        pipe: Loaded StableDiffusionInpaintPipeline.
        image: Original RGB image (PIL).
        mask: Mask image (PIL), white=keep, black=inpaint.
        prompt: Optional text prompt to guide inpainting.

    Returns:
        Inpainted PIL.Image.
    """
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return result
