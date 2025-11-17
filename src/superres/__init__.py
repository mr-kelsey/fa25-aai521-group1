"""
__init__.py

Package initializer for the `superres` module.

This file defines the public API of the package by exposing the most
commonly used functions from submodules:

- enhance_image: Core entry point for image enhancement (single_image.py)
- load_ldm_model: Utility to load a pretrained LDM pipeline (model_utils.py)
- superresolve_image: Run LDM super-resolution on a single image (model_utils.py)
- show_before_after: Visualization helper for side-by-side comparison (viz_utils.py)

"""

from .single_image import enhance_image
from .model_utils import load_ldm_model, superresolve_image
from .viz_utils import show_before_after

__all__ = [
    "enhance_image",
    "load_ldm_model",
    "superresolve_image",
    "show_before_after",
]
