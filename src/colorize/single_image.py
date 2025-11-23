"""Single-image colorization pipeline using a Stable Diffusion img2img model.

Loads a grayscale image from the `data/grey` folder, runs a Stable Diffusion
img2img colorization step, optionally saves and visualizes the result.
"""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .model_utils import load_colorize_model, run_colorization
from .viz_utils import show_before_after


def _load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def colorize_image(
    grey_image_path: str,
    truth_image_path: Optional[str] = None,
    save_to: Optional[str] = None,
    visualize: bool = False,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    prompt: str = "colorized photo, realistic colors, detailed, sharp",
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
    """Colorize a single image using a Stable Diffusion img2img pipeline.

    Args:
        grey_image_path: Path to grayscale input image.
        truth_image_path: Optional path to corresponding ground-truth color image
            used only for visualization/comparison.
        save_to: Base path to save outputs.
        visualize: Show before/after (and optionally ground truth) comparison.
        model_id: Hugging Face model id for the img2img pipeline.
        prompt: Text prompt guiding colorization.
        strength: Img2img strength.
        guidance_scale: Guidance scale for classifier-free guidance.
        num_inference_steps: Number of diffusion steps.

    Returns:
        (grey_pil, colorized_pil, truth_pil or None)
    """
    grey_pil = _load_image_pil(grey_image_path)
    truth_pil = _load_image_pil(truth_image_path) if truth_image_path else None

    pipe = load_colorize_model(model_id=model_id)
    colorized_pil = run_colorization(
        pipe,
        image=grey_pil,
        prompt=prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        grey_pil.save(base.with_name(base.stem + "_input.png"))
        colorized_pil.save(base.with_name(base.stem + "_colorized.png"))
        if truth_pil is not None:
            truth_pil.save(base.with_name(base.stem + "_truth.png"))

    if visualize:
        show_before_after(grey_pil, colorized_pil, title_left="Grayscale", title_right="Colorized")

    return grey_pil, colorized_pil, truth_pil
