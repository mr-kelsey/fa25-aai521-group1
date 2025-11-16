"""
model_utils.py

Utility functions for loading and running Latent Diffusion Models (LDM)
for single-image super-resolution. This module provides:

- `load_ldm_model`: Load a pretrained LDM super-resolution pipeline
  from Hugging Face Diffusers, configured for GPU or CPU.
- `superresolve_image`: Run the pipeline on a single image, returning
  both the low-resolution input and the super-resolved output.

These functions are used by the main pipeline (`single_image.py`) to
support the `--method ldm` option.
"""

import torch
from PIL import Image
from diffusers import LDMSuperResolutionPipeline


def load_ldm_model(model_id: str = "CompVis/ldm-super-resolution-4x-openimages") -> LDMSuperResolutionPipeline:
    """
    Load a Latent Diffusion Model (LDM) super-resolution pipeline.

    Args:
        model_id (str): Hugging Face model identifier. Defaults to
            "CompVis/ldm-super-resolution-4x-openimages".

    Returns:
        LDMSuperResolutionPipeline: A pipeline object ready to run
        super-resolution inference on images.

    Notes:
        - If CUDA is available, the model is loaded on GPU with float16
          precision for efficiency.
        - Otherwise, the model runs on CPU with float32 precision.
    """
    # Select device: GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use half precision on GPU for faster inference, full precision on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load pretrained pipeline from Hugging Face Diffusers
    pipe = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Move pipeline to the selected device
    return pipe.to(device)


def superresolve_image(pipe: LDMSuperResolutionPipeline, image_path: str) -> tuple[Image.Image, Image.Image]:
    """
    Run LDM super-resolution on a single image.

    Args:
        pipe (LDMSuperResolutionPipeline): A loaded pipeline object.
        image_path (str): Path to the input image file.

    Returns:
        tuple[Image.Image, Image.Image]:
            - Low-resolution input image (PIL.Image, resized to 128x128).
            - Super-resolved output image (PIL.Image).

    Notes:
        - The input image is always resized to 128x128 before inference,
          as required by the pretrained LDM model.
        - The pipeline returns a list of images; we take the first one.
    """
    # Load image, convert to RGB, and resize to 128x128
    lr_img = Image.open(image_path).convert("RGB").resize((128, 128))

    # Run the pipeline to generate a super-resolved image
    sr_img = pipe(lr_img).images[0]

    return lr_img, sr_img
