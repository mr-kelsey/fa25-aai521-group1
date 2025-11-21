"""
inpaint_pipeline.py

Command-line entry point for the inpainting pipeline.

Usage example:
    py inpaint_pipeline.py --image data/damaged/0006_d.png --method sd --save-to outputs/0006_inpaint.png
"""

import argparse
from src.inpaint.single_image import inpaint_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image inpainting pipeline")

    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--method", type=str, choices=["sd", "completion"], default="sd",
                        help="Enhancement method: 'sd' (Stable Diffusion Inpainting) or 'completion'")
    parser.add_argument("--model-id", type=str,
                        default="stabilityai/stable-diffusion-inpainting",
                        help="Hugging Face model ID for inpainting")
    parser.add_argument("--save-to", type=str, default="outputs/result.png",
                        help="Base path to save enhanced image")
    parser.add_argument("--visualize", action="store_true", help="Show before/after comparison")
    parser.add_argument("--save-only", action="store_true", help="Save output without visualization")

    args = parser.parse_args()
    visualize = args.visualize and not args.save_only

    orig_img, inpainted_img = inpaint_image(
        image_path=args.image,
        method=args.method,
        model_id=args.model_id,
        save_to=args.save_to,
        visualize=visualize,
    )

    print("Inpainting complete.")
    if args.save_to:
        base = args.save_to.replace(".png", "")
        print(f"Saved: {base}_input.png and {base}_inpainted.png")


if __name__ == "__main__":
    main()
