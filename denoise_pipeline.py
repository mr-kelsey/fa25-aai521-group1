"""Command-line entry point for the denoising pipeline.

Usage example:
    python denoise_pipeline.py --image data/noisy/0006_n.png --save-to outputs/0006_denoised.png
"""

import argparse

from src.denoise.single_image import denoise_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image denoising pipeline")

    parser.add_argument("--image", type=str, required=True, help="Path to noisy input image")
    parser.add_argument("--model-id", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Hugging Face model ID for denoising (img2img)")
    parser.add_argument("--save-to", type=str, default="outputs/denoised_result.png",
                        help="Base path to save enhanced image")
    parser.add_argument("--visualize", action="store_true", help="Show before/after comparison")
    parser.add_argument("--save-only", action="store_true", help="Save output without visualization")
    parser.add_argument("--prompt", type=str,
                        default="high quality clean photo, detailed, sharp",
                        help="Optional text prompt to guide denoising")
    parser.add_argument("--strength", type=float, default=0.4,
                        help="Img2img strength controlling how much the model changes the input")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num-inference-steps", type=int, default=30,
                        help="Number of diffusion steps")

    args = parser.parse_args()
    visualize = args.visualize and not args.save_only

    noisy_img, denoised_img = denoise_image(
        image_path=args.image,
        model_id=args.model_id,
        save_to=args.save_to,
        visualize=visualize,
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    print("Denoising complete.")
    if args.save_to:
        base = args.save_to.replace(".png", "")
        print(f"Saved: {base}_input.png and {base}_denoised.png")


if __name__ == "__main__":
    main()
