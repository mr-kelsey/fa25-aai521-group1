"""Single-image denoising pipeline."""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .model_utils import load_denoise_model, run_denoising
from src.inpaint.viz_utils import show_before_after


def _load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def denoise_image(
    image_path: str,
    model_id: str = None,
    save_to: Optional[str] = None,
    visualize: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """Denoise a single image and optionally save/visualize the result."""
    noisy_pil = _load_image_pil(image_path)

    model = load_denoise_model(model_id=model_id)
    denoised_pil = run_denoising(model, image=noisy_pil)

    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        noisy_pil.save(base.with_name(base.stem + "_input.png"))
        denoised_pil.save(base.with_name(base.stem + "_denoised.png"))

    if visualize:
        show_before_after(noisy_pil, denoised_pil, title_left="Noisy", title_right="Denoised")

    return noisy_pil, denoised_pil
