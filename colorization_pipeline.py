"""Command-line entry point for the image colorization pipeline."""

import argparse
from src.colorize.single_image import colorize_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Run image colorization")

    parser.add_argument(
        "--grey-image",
        type=str,
        required=True,
        help="Path to grayscale input image",
    )

    parser.add_argument(
        "--truth-image",
        type=str,
        required=False,
        help="Optional path to ground-truth color image",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="caffe",
        help="Model to use for colorization (default: 'caffe')",
    )

    parser.add_argument(
        "--save-to",
        type=str,
        default="outputs/colorized_result.png",
        help="Path to save the colorized image",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show before/after comparison window",
    )

    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save output without visualization",
    )

    args = parser.parse_args()

    visualize = args.visualize and not args.save_only

    print(f"\nRunning colorization with model: {args.model_id}...\n")

    _, _, _ = colorize_image(
        grey_image_path=args.grey_image,
        truth_image_path=args.truth_image,
        save_to=args.save_to,
        visualize=visualize,
        model_id=args.model_id,
    )

    print("Colorization complete!")

    if args.save_to:
        base_name = args.save_to.rsplit('.', 1)[0]
        print(f"Outputs saved with base name: {base_name}")

if __name__ == "__main__":
    main()
