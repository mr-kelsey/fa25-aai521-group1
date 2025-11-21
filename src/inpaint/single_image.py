"""
single_image.py

Core inpainting logic: loads image, prepares/loads mask, runs model,
optionally saves and visualizes.
"""

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import cv2
import numpy as np

from .model_utils import load_inpaint_model, run_inpainting
from .viz_utils import show_before_after

# teammate helper imports (already exist in notebooks/helper)
from notebooks.helper.completion import Completion


def _load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _ensure_binary_mask(mask_img: Image.Image) -> Image.Image:
    arr = np.array(mask_img.convert("L"))
    bin_arr = (arr >= 128).astype(np.uint8) * 255
    return Image.fromarray(bin_arr, mode="L")


def _generate_mask_from_completion(image_bgr: np.ndarray) -> Image.Image:
    """
    Generate a binary mask by thresholding bright specks in the image.
    White (255) = region to inpaint, Black (0) = keep.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Threshold: pixels brighter than 240 are considered "damage"
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask, mode="L")


def inpaint_image(
    image_path: str,
    method: str = "sd",
    model_id: str = "runwayml/stable-diffusion-inpainting",
    save_to: Optional[str] = None,
    visualize: bool = False,
    prompt: str = "",
) -> Tuple[Image.Image, Image.Image]:
    """
    Inpaint a single image using Stable Diffusion Inpainting or Completion.

    Args:
        image_path: Path to original image.
        method: 'sd' for Stable Diffusion Inpainting, 'completion' for patch-based Completion.
        model_id: Hugging Face model id.
        save_to: Base path to save outputs.
        visualize: Show before/after comparison.
        prompt: Optional text prompt to guide inpainting content.

    Returns:
        (original_image_pil, inpainted_image_pil)
    """
    orig_pil = _load_image_pil(image_path)

    if method == "sd":
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        mask_pil = _generate_mask_from_completion(img_bgr)
        pipe = load_inpaint_model(model_id=model_id)
        inpainted_pil = run_inpainting(pipe, image=orig_pil, mask=mask_pil, prompt=prompt)

    elif method == "completion":
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        comp = Completion(image=img_bgr, pipeline=lambda p, m: (p, m), input_shape=(100, 100))
        inpainted_pil = Image.fromarray(cv2.cvtColor(comp.generated_image, cv2.COLOR_BGR2RGB))

    else:
        raise ValueError("method must be 'sd' or 'completion'")

    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        orig_pil.save(base.with_name(base.stem + "_input.png"))
        inpainted_pil.save(base.with_name(base.stem + "_inpainted.png"))

    if visualize:
        show_before_after(orig_pil, inpainted_pil, title_left="Original", title_right="Inpainted")

    return orig_pil, inpainted_pil
