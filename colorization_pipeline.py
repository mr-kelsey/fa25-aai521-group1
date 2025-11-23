"""Command-line entry point for the colorization pipeline.

"""

import argparse

from src.colorize.single_image import colorize_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image colorization pipeline")

    parser.add_argument("--grey-image", type=str, required=True,
                        help="Path to grayscale input image (e.g., data/grey/0006_g.png)")
    parser.add_argument("--truth-image", type=str, required=False,
                        help="Optional path to corresponding ground-truth color image (for comparison)")
    parser.add_argument("--model-id", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model ID for colorization (img2img)")
    parser.add_argument("--save-to", type=str, default="outputs/colorized_result.png",
                        help="Base path to save enhanced image")
    parser.add_argument("--visualize", action="store_true", help="Show before/after comparison")
    parser.add_argument("--save-only", action="store_true", help="Save output without visualization")
    parser.add_argument("--prompt", type=str,
                        default="colorized photo, realistic colors, detailed, sharp",
                        help="Optional text prompt to guide colorization")
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Img2img strength controlling how much the model changes the input")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num-inference-steps", type=int, default=30,
                        help="Number of diffusion steps")

    args = parser.parse_args()
    visualize = args.visualize and not args.save_only

    grey_img, colorized_img, truth_img = colorize_image(
        grey_image_path=args.grey_image,
        truth_image_path=args.truth_image,
        save_to=args.save_to,
        visualize=visualize,
        model_id=args.model_id,
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    print("Colorization complete.")
    if args.save_to:
        base = args.save_to.replace(".png", "")
        suffix = f" and {base}_truth.png" if args.truth_image else ""
        print(f"Saved: {base}_input.png and {base}_colorized.png{suffix}")


if __name__ == "__main__":
    main()
