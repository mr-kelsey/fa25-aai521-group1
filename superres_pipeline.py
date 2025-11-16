"""
superres_pipeline.py

Command-line entry point for the super-resolution pipeline.

This script allows users to run image enhancement using either:

- Latent Diffusion Model (LDM) super-resolution
- Patch-based Completion method (team contribution)

It provides CLI arguments for selecting the method, patch size,
model ID, output path, and visualization options.

Usage example:
    py superres_pipeline.py --image data/scaled/0006_x4.png --method ldm --save-to outputs/0006_ldm.png
"""

import argparse
from src.superres.single_image import enhance_image


def main() -> None:
    """
    Parse command-line arguments and run the image enhancement pipeline.

    CLI Arguments:
        --image (str): Path to input image (required).
        --method (str): Enhancement method ("ldm" or "completion").
        --completion-patch (int, int): Patch size for completion method.
        --model-id (str): Hugging Face model ID for LDM.
        --save-to (str): Base path to save enhanced image.
        --visualize (flag): Show before/after comparison in a matplotlib window.
        --save-only (flag): Save outputs without visualization.

    Behavior:
        - If --save-only is passed, visualization is disabled even if --visualize is set.
        - Saves both input and enhanced images with suffixes "_input.png" and "_enhanced.png".
        - Prints confirmation messages after enhancement and saving.
    """
    parser = argparse.ArgumentParser(description="Run image enhancement pipeline")

    # Required input image path
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    # Choice of enhancement method
    parser.add_argument("--method", type=str, choices=["ldm", "completion"], default="ldm",
                        help="Enhancement method")

    # Patch size for completion method (two integers: height width)
    parser.add_argument("--completion-patch", type=int, nargs=2, default=[100, 100],
                        help="Patch size for completion method")

    # Hugging Face model ID for LDM
    parser.add_argument("--model-id", type=str,
                        default="CompVis/ldm-super-resolution-4x-openimages",
                        help="Hugging Face model ID for LDM")

    # Output path for saving results
    parser.add_argument("--save-to", type=str, default="outputs/result.png",
                        help="Base path to save enhanced image")

    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Show before/after comparison")
    parser.add_argument("--save-only", action="store_true",
                        help="Save output without visualization")

    args = parser.parse_args()

    # If --save-only is passed, override visualization
    visualize = args.visualize and not args.save_only

    # Run the enhancement pipeline
    lr_img, sr_img = enhance_image(
        image_path=args.image,
        method=args.method,
        completion_patch=tuple(args.completion_patch),
        model_id=args.model_id,
        save_to=args.save_to,
        visualize=visualize
    )

    # Print confirmation messages
    print("Enhancement complete.")
    if args.save_to:
        print(f"Saved: {args.save_to.replace('.png','')}_input.png "
              f"and {args.save_to.replace('.png','')}_enhanced.png")


if __name__ == "__main__":
    main()
