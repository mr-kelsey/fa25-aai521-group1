"""
Unified Model Manager for AI-Powered Image Enhancement Suite

This module provides a centralized way to manage and load various AI models
used in the image enhancement application, including caching functionality
to avoid reloading models unnecessarily.
"""

import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline
import torchvision.transforms as transforms
import os
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
    Apply denoising using Hugging Face diffusion model.
    
    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)
        
    Returns:
        numpy.ndarray: Denoised image in OpenCV format (BGR)
    """
    # Convert OpenCV image (BGR) to PIL (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Create a mask for the entire image (we want to denoise everything)
    mask = Image.new('L', pil_image.size, 255)  # White mask means denoise everything
    
    # Load the model
    model = model_manager.get_model("denoising")
    if model is None:
        # Fallback to basic Gaussian blur if model loading failed
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    # Perform denoising/inpainting
    denoised_pil = model(
        prompt="clean, clear, high quality image",
        image=pil_image,
        mask_image=mask,
        num_inference_steps=20,
        strength=0.75
    ).images[0]
    
    # Convert back to OpenCV format (RGB to BGR)
    denoised_np = np.array(denoised_pil)
    denoised_cv = cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR)
    
    # Ensure the denoised image has the same dimensions as the input
    if denoised_cv.shape != image.shape:
        denoised_cv = cv2.resize(denoised_cv, (image.shape[1], image.shape[0]))
    
    return denoised_cv


def transformer_super_resolution(image):
    """
    Apply transformer-based super resolution to the image.
    
    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)
        
    Returns:
        numpy.ndarray: Upscaled image in OpenCV format (BGR)
    """
    # Load the model
    model = model_manager.get_model("super_resolution")
    
    if model is None:
        # Fallback to basic upscaling if model loading failed
        height, width = image.shape[:2]
        return cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # Convert OpenCV image (HWC, BGR) to PIL (HWC, RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Super resolution using Real-ESRGAN
    # Convert PIL image to tensor
    img_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to(model.device)
    
    # Normalize the image
    img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    
    # Perform super resolution
    with torch.no_grad():
        output = model(img_tensor)
    
    # Denormalize
    output = (output + 1.0) / 2.0  # Convert back to [0, 1]
    
    # Convert tensor to numpy array
    output = output.squeeze().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    
    # Convert back to OpenCV format (RGB to BGR)
    output_cv = cv2.cvtColor((output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return output_cv


def neural_colorization(image):
    """
    Apply neural network-based colorization to the grayscale image.
    
    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)
        
    Returns:
        numpy.ndarray: Colorized image in OpenCV format (BGR)
    """
    # Load the model
    model = model_manager.get_model("colorization")
    
    if model is None:
        # Fallback to basic grayscale if model loading failed
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    
    # Convert OpenCV image (HWC, BGR) to PIL (HWC, RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Resize image to model's expected size (most colorization models expect 224x224)
    pil_image = pil_image.resize((224, 224))
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2LAB)
    img_l = img_lab[:, :, 0]  # Extract L channel
    
    # Normalize the L channel to [0, 1] and then to [-1, 1] as expected by many models
    img_l_norm = (img_l.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # Create tensor from L channel
    img_tensor = torch.tensor(img_l_norm).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Move to device where model is
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Run the model to get AB channels
    with torch.no_grad():
        output_ab = model(img_tensor)
    
    # Convert back to numpy and scale
    output_ab = output_ab.squeeze().cpu().numpy()
    
    # Scale back AB channels from [-1, 1] to [0, 255] range
    output_ab = (output_ab + 1) * 127.5
    output_ab = np.clip(output_ab, 0, 255)
    
    # Convert original L back to [0, 255] range
    img_l_rescaled = (img_l_norm + 1) * 127.5
    img_l_rescaled = np.clip(img_l_rescaled, 0, 255)
    
    # Stack L with AB to get full LAB image
    img_lab_resized = np.zeros((224, 224, 3), dtype=np.uint8)
    img_lab_resized[:, :, 0] = img_l_rescaled.astype(np.uint8)  # L channel
    img_lab_resized[:, :, 1:] = output_ab.transpose(1, 2, 0).astype(np.uint8)  # AB channels
    
    # Convert LAB back to RGB
    colorized_rgb = cv2.cvtColor(img_lab_resized, cv2.COLOR_LAB2RGB)
    
    # Convert back to BGR for consistency with the rest of the code
    colorized_bgr = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)
    
    # Resize back to original size
    original_height, original_width = image.shape[:2]
    colorized_bgr = cv2.resize(colorized_bgr, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    
    return colorized_bgr


def neural_inpainting(image):
    """
    Apply neural network-based inpainting to restore damaged regions in the image.
    
    Args:
        image (numpy.ndarray): Input image in OpenCV format (BGR)
        
    Returns:
        numpy.ndarray: Inpainted image in OpenCV format (BGR)
    """
    # Load the model
    model = model_manager.get_model("inpainting")
    
    if model is None:
        # Fallback to basic inpainting if model loading failed
        # Create a mask with some region removed
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

        # Simply fill the masked region with average of the surroundings
        result = image.copy()
        mean_color = np.mean(image[mask==0], axis=0)  # Average of non-masked pixels
        result[mask==255] = mean_color  # Fill masked region with average
        
        return result
    
    # Convert OpenCV image (BGR) to PIL (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Create a mask for the region to inpaint
    mask = Image.new('L', pil_image.size, 0)  # Black background
    # Create a white rectangle in the center to indicate what to inpaint 
    mask_draw = Image.new('L', pil_image.size, 0)
    draw = ImageDraw.Draw(mask_draw)
    w, h = pil_image.size
    draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
    mask = mask_draw

    # Perform inpainting
    try:
        inpainted_pil = model(
            prompt="complete, restored, clean image",
            image=pil_image,
            mask_image=mask,
            num_inference_steps=20,
            strength=0.75
        ).images[0]
        
        # Convert back to OpenCV format (RGB to BGR)
        inpainted_np = np.array(inpainted_pil)
        inpainted_cv = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)
        
        # Ensure the inpainted image has the same dimensions as the input
        if inpainted_cv.shape != image.shape:
            inpainted_cv = cv2.resize(inpainted_cv, (image.shape[1], image.shape[0]))
        
        return inpainted_cv
    except Exception as e:
        print(f"Inpainting process failed: {e}")
        # Fallback to basic inpainting
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        
        result = image.copy()
        mean_color = np.mean(image[mask==0], axis=0)
        result[mask==255] = mean_color
        
        return result