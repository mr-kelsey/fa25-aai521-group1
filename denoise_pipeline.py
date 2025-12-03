"""Command-line entry point for the denoising pipeline.

This pipeline uses a UNet2DModel-based denoiser (default:
`skytnt/anime-denoiser`) via Diffusers to perform direct image denoising.
"""

import argparse

from src.denoise.single_image import denoise_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image denoising pipeline")

    parser.add_argument("--image", type=str, required=True, help="Path to noisy input image")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model ID for denoising (Diffusers UNet2DModel)",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default="outputs/denoised_result.png",
        help="Base path to save enhanced image",
    )
    parser.add_argument("--visualize", action="store_true", help="Show before/after comparison")
    parser.add_argument("--save-only", action="store_true", help="Save output without visualization")

    args = parser.parse_args()
    visualize = args.visualize and not args.save_only

    noisy_img, denoised_img = denoise_image(
        image_path=args.image,
        model_id=args.model_id,
        save_to=args.save_to,
        visualize=visualize,
    )

    print("Denoising complete.")
    if args.save_to:
        base = args.save_to.replace(".png", "")
        print(f"Saved: {base}_input.png and {base}_denoised.png")


if __name__ == "__main__":
    main()
