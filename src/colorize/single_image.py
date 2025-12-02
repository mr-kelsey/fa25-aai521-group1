# src/colorize/single_image.py

from pathlib import Path
from PIL import Image
from .model_utils import load_colorize_model, run_colorization
from .viz_utils import show_before_after

def colorize_image(
    grey_image_path: str,
    truth_image_path: str = None,
    save_to: str = None,
    visualize: bool = False,
    model_id: str = "caffe",
):
    """
    Loads a grayscale image, colorizes it using the specified model,
    and optionally saves and visualizes the result.
    """
    # Load the model using the centralized loader
    model = load_colorize_model(model_id)

    # Open the grayscale image
    grey_pil = Image.open(grey_image_path).convert("RGB")

    # Run colorization
    colorized_pil = run_colorization(model, grey_pil)

    # Load ground truth if provided
    truth_pil = Image.open(truth_image_path).convert("RGB") if truth_image_path else None

    # Save outputs if a path is specified
    if save_to:
        base = Path(save_to)
        base.parent.mkdir(parents=True, exist_ok=True)
        
        # Save a copy of the input image for comparison
        input_save_path = base.with_name(f"{base.stem}_input.png")
        grey_pil.save(input_save_path)
        
        # Save the main colorized output
        colorized_pil.save(save_to)

        # Save ground truth if it exists
        if truth_pil:
            truth_save_path = base.with_name(f"{base.stem}_truth.png")
            truth_pil.save(truth_save_path)

    # Visualize if requested
    if visualize:
        show_before_after(grey_pil, colorized_pil, truth_pil)

    return grey_pil, colorized_pil, truth_pil