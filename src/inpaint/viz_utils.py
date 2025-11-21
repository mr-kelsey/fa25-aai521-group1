"""
viz_utils.py

Visualization helpers for side-by-side comparisons.
"""

import matplotlib.pyplot as plt
from PIL import Image


def show_before_after(
    left_img: Image.Image,
    right_img: Image.Image,
    title_left: str = "Original",
    title_right: str = "Inpainted",
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(left_img)
    axs[0].set_title(title_left)
    axs[0].axis("off")

    axs[1].imshow(right_img)
    axs[1].set_title(title_right)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
