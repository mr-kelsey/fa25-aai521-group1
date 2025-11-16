"""
single_image.py

Core enhancement logic for the super-resolution pipeline.

This module provides the `enhance_image` function, which supports two
methods of image enhancement:

1. LDM (Latent Diffusion Model) super-resolution:
   - Uses Hugging Face Diffusers to upscale images.
   - Input images are resized to 128x128 before inference.

2. Completion (team contribution):
   - Uses the full Completion class for patch-based enhancement.
   - Handles patch extraction, upscaling, stitching, and visualization.

The function returns both the original (low-resolution) and enhanced
images as PIL.Image objects, and optionally saves them to disk.
"""

import cv2
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from .model_utils import load_ldm_model, superresolve_image
from .viz_utils import show_before_after

# teammate helper imports
from notebooks.helper.completion import Completion
from notebooks.helper.utils import add_damage, add_noise, change_scale, remove_color


def enhance_image(
    image_path: str,
    method: str = "ldm",
    completion_patch: Tuple[int, int] = (100, 100),
    model_id: str = "CompVis/ldm-super-resolution-4x-openimages",
    save_to: Optional[str] = None,
    visualize: bool = False
) -> Tuple[Image.Image, Image.Image]:
    """
    Enhance a single image using either LDM or Completion.

    Args:
        image_path (str): Path to the input image file.
        method (str): Enhancement method to use.
            - "ldm": Latent Diffusion Model super-resolution.
            - "completion": Patch-based Completion class.
        completion_patch (Tuple[int, int]): Patch size for Completion
            method (height, width). Default is (100, 100).
        model_id (str): Hugging Face model ID for LDM. Default is
            "CompVis/ldm-super-resolution-4x-openimages".
        save_to (Optional[str]): Base path for saving outputs. If None,
            results are not saved.
        visualize (bool): If True, show before/after comparison in a
            matplotlib window.

    Returns:
        Tuple[Image.Image, Image.Image]:
            - Low-resolution input image (PIL.Image).
            - Enhanced output image (PIL.Image).

    Raises:
        FileNotFoundError: If the input image cannot be loaded.
        ValueError: If an unsupported method is specified.
    """
    if method == "ldm":
        # Load pretrained LDM pipeline
        pipe = load_ldm_model(model_id=model_id)

        # Run super-resolution on the input image
        lr_img, sr_img = superresolve_image(pipe, image_path)

    elif method == "completion":
        # Load image in BGR format (OpenCV default)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Define patch processing function for Completion
        def patch_upscale(patch, attention_mask):
            # Upscale patch by factor of 3 using bicubic interpolation
            patch = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

            # Upscale attention mask and clip values to [0, 1]
            attention_mask = cv2.resize(attention_mask, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            attention_mask = np.clip(attention_mask, 0, 1)
            return patch, attention_mask

        # Instantiate the full Completion class
        comp = Completion(image=img_bgr, pipeline=patch_upscale, input_shape=completion_patch)

        # Convert generated image (BGR → RGB) to PIL.Image
        sr_img = Image.fromarray(cv2.cvtColor(comp.generated_image, cv2.COLOR_BGR2RGB))

        # Convert original image (BGR → RGB) to PIL.Image, resized to 128x128
        lr_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((128, 128))

    else:
        raise ValueError("method must be 'ldm' or 'completion'")

    # Save both input and enhanced images if requested
    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        lr_img.save(base.with_name(base.stem + "_input.png"))
        sr_img.save(base.with_name(base.stem + "_enhanced.png"))

    # Optionally show side-by-side visualization
    if visualize:
        show_before_after(lr_img, sr_img, title_left="Input", title_right="Enhanced")

    return lr_img, sr_img
