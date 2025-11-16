"""
viz_utils.py

Visualization utilities for the super-resolution pipeline.

This module provides helper functions to display images side-by-side
for comparison. It is primarily used by `single_image.py` when the
`--visualize` flag is enabled.

Functions:
    show_before_after(lr_img, sr_img, title_left="Input", title_right="Enhanced"):
        Display a side-by-side comparison of the input (low-resolution)
        and enhanced (super-resolved) images.
"""

import matplotlib.pyplot as plt
from PIL import Image


def show_before_after(
    lr_img: Image.Image,
    sr_img: Image.Image,
    title_left: str = "Input",
    title_right: str = "Enhanced"
) -> None:
    """
    Display side-by-side comparison of input and enhanced images.

    Args:
        lr_img (PIL.Image.Image): The low-resolution or original image.
        sr_img (PIL.Image.Image): The enhanced or super-resolved image.
        title_left (str): Title for the left subplot. Defaults to "Input".
        title_right (str): Title for the right subplot. Defaults to "Enhanced".

    Returns:
        None. Displays a matplotlib figure window with the comparison.

    Notes:
        - Axes are hidden to emphasize the image content.
        - The figure is sized to 10x5 inches for readability.
        - `plt.tight_layout()` ensures titles and images do not overlap.
    """
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Show the low-resolution image on the left
    axs[0].imshow(lr_img)
    axs[0].set_title(title_left)
    axs[0].axis("off")  # Hide axes for cleaner display

    # Show the super-resolved image on the right
    axs[1].imshow(sr_img)
    axs[1].set_title(title_right)
    axs[1].axis("off")  # Hide axes for cleaner display

    # Adjust layout to prevent overlap and show the figure
    plt.tight_layout()
    plt.show()
