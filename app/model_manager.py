"""
Unified Model Manager for AI-Powered Image Enhancement Suite

This module provides a centralized way to manage and load various AI models
used in the image enhancement application, including caching functionality
to avoid reloading models unnecessarily.
"""

import sys
import os
# Add the project root directory to the Python path to allow imports from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


class ModelManager:
    """
    Unified model manager for loading and caching all AI models used in the application.
    """

    def __init__(self):
        """
        Initialize the model manager with all supported models.
        """
        self.models = {}
        # Use CUDA if available, but check for compatibility
        if torch.cuda.is_available():
            # Check if the GPU is compatible with the current PyTorch installation
            try:
                # Try to get capabilities to check compatibility
                self.device = torch.cuda.current_device()
                self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            except:
                # If there are compatibility issues, fall back to CPU
                self.device = "cpu"
                self.torch_dtype = torch.float32
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        print(f"ModelManager: Using device {self.device}, dtype {self.torch_dtype}")

    def load_model(self, model_name):
        """
        Load a specific model by name if not already loaded.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            The loaded model or None if loading failed
        """
        if model_name in self.models:
            return self.models[model_name]
        
        model = None
        try:
            if model_name == "inpainting" or model_name == "denoising":
                # Using the same model for both inpainting and denoising
                model = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=self.torch_dtype
                )
                model = model.to(self.device)
                
            elif model_name == "super_resolution":
                # Load Real-ESRGAN for super resolution
                model = torch.hub.load("xinntao/Real-ESRGAN", "RealESRGAN_x4plus", 
                                       pretrained=True, trust_repo=True)
                model = model.to(self.device)
                model.eval()
                
            elif model_name == "colorization":
                # Load colorization model
                model = torch.hub.load('junyanz/pytorch-CycleGAN-and-pix2pix', 
                                       'colorization', pretrained=True)
                model = model.to(self.device)
                model.eval()
            
            if model:
                self.models[model_name] = model
                return model
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            return None
    
        return None

    def get_model(self, model_name):
        """
        Get a model from cache or load it if not already loaded.
        
        Args:
            model_name (str): Name of the model to retrieve
            
        Returns:
            The model instance or None if not available
        """
        if model_name not in self.models:
            return self.load_model(model_name)
        return self.models[model_name]

    def clear_cache(self, model_name=None):
        """
        Clear model cache, either for a specific model or all models.
        
        Args:
            model_name (str, optional): Name of the model to clear. If None, clear all models.
        """
        if model_name and model_name in self.models:
            del self.models[model_name]
        elif model_name is None:
            self.models.clear()

    def get_available_models(self):
        """
        Get a list of available models.
        
        Returns:
            list: List of model names
        """
        return list(self.models.keys())


# Global model manager instance
model_manager = ModelManager()


def huggingface_denoise(image):
    """
    Apply denoising using Hugging Face diffusion model via the new pipeline.

    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)

    Returns:
        numpy.ndarray: Denoised image in OpenCV format (BGR)
    """
    try:
        from src.denoise.single_image import denoise_image
        import tempfile
        import os

        # Create a temporary file to pass to the pipeline
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Convert the image from OpenCV (BGR) to PIL (RGB) and save to temp file
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Use the new denoise pipeline with optimized parameters
            noisy_pil, denoised_pil = denoise_image(
                image_path=temp_path,
                model_id="runwayml/stable-diffusion-v1-5",
                visualize=False,  # Don't visualize during processing
                strength=0.4,
                guidance_scale=7.5,
                num_inference_steps=20  # Reduced steps for faster processing
            )

            # Convert the result (PIL Image) back to OpenCV format (RGB to BGR)
            denoised_np = np.array(denoised_pil)
            denoised_cv = cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR)

            # Ensure the denoised image has the same dimensions as the input
            if denoised_cv.shape != image.shape:
                denoised_cv = cv2.resize(denoised_cv, (image.shape[1], image.shape[0]))

            return denoised_cv

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except ImportError as e:
        print(f"Warning: Could not import denoising pipeline: {e}")
        # Fallback to basic Gaussian blur
        return cv2.GaussianBlur(image, (5, 5), 0)
    except Exception as e:
        print(f"Error using denoising pipeline: {e}")
        # Fallback to basic Gaussian blur
        return cv2.GaussianBlur(image, (5, 5), 0)


def transformer_super_resolution(image):
    """
    Apply transformer-based super resolution to the image using the new superres pipeline.

    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)

    Returns:
        numpy.ndarray: Upscaled image in OpenCV format (BGR)
    """
    try:
        from src.superres.single_image import enhance_image
        import tempfile
        import os

        # Create a temporary file to pass to the pipeline
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Convert the image from OpenCV (BGR) to PIL (RGB) and save to temp file
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Use the new superres pipeline with LDM method for super resolution
            _, enhanced_img = enhance_image(
                image_path=temp_path,
                method="ldm",  # Use Latent Diffusion Model
                model_id="CompVis/ldm-super-resolution-4x-openimages",  # Default model
                visualize=False  # Don't visualize during processing
            )

            # Convert the result (PIL Image) back to OpenCV format (RGB to BGR)
            enhanced_np = np.array(enhanced_img)
            enhanced_cv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

            return enhanced_cv

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except ImportError as e:
        print(f"Error: Could not import super-resolution pipeline: {e}")
        raise
    except Exception as e:
        print(f"Error using super resolution pipeline: {e}")
        raise


def neural_colorization(image):
    """
    Apply neural network-based colorization using the new colorization pipeline.

    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)

    Returns:
        numpy.ndarray: Colorized image in OpenCV format (BGR)
    """
    try:
        from src.colorize.single_image import colorize_image
        import tempfile
        import os

        # Create a temporary file to pass to the pipeline
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Convert the image from OpenCV (BGR) to PIL (RGB) and save to temp file
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Use the new colorization pipeline
            grey_pil, colorized_pil, _ = colorize_image(
                grey_image_path=temp_path,
                visualize=False,  # Don't visualize during processing
                model_id="runwayml/stable-diffusion-v1-5",
                prompt="colorized photo, realistic colors, detailed, sharp",
                strength=0.5  # Reduced strength for faster processing
            )

            # Convert the result (PIL Image) back to OpenCV format (RGB to BGR)
            colorized_np = np.array(colorized_pil)
            colorized_cv = cv2.cvtColor(colorized_np, cv2.COLOR_RGB2BGR)

            # Ensure the colorized image has the same dimensions as the input
            if colorized_cv.shape != image.shape:
                colorized_cv = cv2.resize(colorized_cv, (image.shape[1], image.shape[0]))

            return colorized_cv

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except ImportError as e:
        print(f"Error: Could not import colorization pipeline: {e}")
        raise
    except Exception as e:
        print(f"Error using colorization pipeline: {e}")
        raise


def neural_inpainting(image):
    """
    Apply neural network-based inpainting using the new inpainting pipeline.

    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)

    Returns:
        numpy.ndarray: Inpainted image in OpenCV format (BGR)
    """
    try:
        from src.inpaint.single_image import inpaint_image
        import tempfile
        import os

        # Create a temporary file to pass to the pipeline
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Convert the image from OpenCV (BGR) to PIL (RGB) and save to temp file
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Use the new inpainting pipeline
            orig_pil, inpainted_pil = inpaint_image(
                image_path=temp_path,
                method="sd",  # Use Stable Diffusion Inpainting
                model_id="runwayml/stable-diffusion-inpainting",
                visualize=False,  # Don't visualize during processing
                prompt="complete, restored, clean image"
            )

            # Convert the result (PIL Image) back to OpenCV format (RGB to BGR)
            inpainted_np = np.array(inpainted_pil)
            inpainted_cv = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)

            # Ensure the inpainted image has the same dimensions as the input
            if inpainted_cv.shape != image.shape:
                inpainted_cv = cv2.resize(inpainted_cv, (image.shape[1], image.shape[0]))

            return inpainted_cv

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except ImportError as e:
        print(f"Error: Could not import inpainting pipeline: {e}")
        raise
    except Exception as e:
        print(f"Error using inpainting pipeline: {e}")
        raise