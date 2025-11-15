from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import uuid
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

# Try to import cv2, but handle the case where it's not available
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not available, some features may be limited")

# Try to import skimage.metrics, but handle the case where it's not available
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None
    print("Warning: scikit-image not available, PSNR/SSIM metrics will be disabled")

# Import for super resolution
import torchvision.transforms as transforms

# Import unified model manager with error handling
try:
    from model_manager import (
        huggingface_denoise,
        transformer_super_resolution,
        neural_colorization,
        neural_inpainting
    )
except ImportError as e:
    print(f"Warning: Could not import model_manager functions: {e}")
    print("Using fallback implementations.")

    # Define fallback functions if imports fail
    def huggingface_denoise(image):
        # Use OpenCV's denoising function as a fallback if cv2 is available
        if cv2 is not None:
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 10, 7, 21)
        else:
            # If cv2 is not available, return image unchanged
            return image.copy()

    def transformer_super_resolution(image):
        # Simple upscaling as a fallback if cv2 is available
        if cv2 is not None:
            original_height, original_width = image.shape[:2]
            new_width = original_width * 2
            new_height = original_height * 2
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            # If cv2 is not available, return image unchanged
            return image.copy()

    def neural_colorization(image):
        # Return image as is for now
        return image.copy()

    def neural_inpainting(image):
        # Return image as is for now
        return image.copy()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_csrf_protection'
app.debug = True  # Enable debug mode to see print statements in console

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload and output directories if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_noise_to_image(image):
    """Add noise to an image"""
    # Convert to float
    img_float = image.astype(np.float32)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape)
    noisy_img = img_float + noise

    # Clip values to valid range
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    if psnr is None:
        # If skimage is not available, return None or a simple calculation
        # PSNR is usually calculated as 20 * log10(max_value) - 10 * log10(MSE)
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        # Use skimage's implementation
        return psnr(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    if ssim is None:
        # If skimage is not available, return None
        # Note: A real implementation would require the SSIM algorithm
        return None
    else:
        # Use skimage's implementation
        # Check if we have cv2 for color conversion, otherwise assume RGB
        if cv2 is not None:
            # Convert BGR to RGB for scikit-image compatibility
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            # Assume RGB format already
            img1_rgb = img1
            img2_rgb = img2

        return ssim(img1_rgb, img2_rgb, channel_axis=-1)

def denoise_image(noisy_img):
    """Apply denoising to the noisy image"""
    try:
        denoised_img = huggingface_denoise(noisy_img)
    except Exception as e:
        print(f"Error in denoising model: {e}")
        # Fallback to a simple denoising method if cv2 is available
        if cv2 is not None:
            denoised_img = cv2.fastNlMeansDenoisingColored(noisy_img, None, 10, 10, 7, 21)
        else:
            # If cv2 isn't available, return the original image
            denoised_img = noisy_img

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(noisy_img, denoised_img)
    ssim_value = calculate_ssim(noisy_img, denoised_img)

    return denoised_img, psnr_value, ssim_value

def enhance_image(img_path, task_type):
    """Apply selected enhancement task to the image"""
    # Use cv2 to load image if available, otherwise use PIL
    if cv2 is not None:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Could not load image with OpenCV")
    else:
        # Use PIL as fallback
        try:
            pil_image = Image.open(img_path)
            image = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Could not load image with PIL: {e}")

    if task_type == 'denoising':
        try:
            # Add noise to the image for demonstration purposes
            noisy_img = add_noise_to_image(image)

            # Apply denoising
            enhanced_img, psnr_val, ssim_val = denoise_image(noisy_img)
        except Exception as e:
            print(f"Error in denoising process: {e}")
            # Fallback: return the original image with error metrics
            enhanced_img = image
            psnr_val = None
            ssim_val = None

        # Save enhanced image using cv2 if available, otherwise PIL
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_denoised{ext}")
        if cv2 is not None:
            cv2.imwrite(output_path, enhanced_img)
        else:
            # Convert numpy array back to PIL Image and save
            enhanced_pil = Image.fromarray(enhanced_img)
            enhanced_pil.save(output_path)

        return output_path, psnr_val, ssim_val

    elif task_type == 'super_resolution':
        try:
            # Super resolution using pre-trained transformer model
            upscaled_img = transformer_super_resolution(image)
        except Exception as e:
            print(f"Error in super resolution process: {e}")
            # Fallback: return the original image
            upscaled_img = image

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_super_res{ext}")
        if cv2 is not None:
            cv2.imwrite(output_path, upscaled_img)
        else:
            # Convert numpy array back to PIL Image and save
            enhanced_pil = Image.fromarray(upscaled_img)
            enhanced_pil.save(output_path)

        return output_path, None, None

    elif task_type == 'colorization':
        try:
            # Neural network-based colorization
            colorized_img = neural_colorization(image)
        except Exception as e:
            print(f"Error in colorization process: {e}")
            # Fallback: return the original image
            colorized_img = image

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_colorized{ext}")
        if cv2 is not None:
            cv2.imwrite(output_path, colorized_img)
        else:
            # Convert numpy array back to PIL Image and save
            enhanced_pil = Image.fromarray(colorized_img)
            enhanced_pil.save(output_path)

        return output_path, None, None

    elif task_type == 'inpainting':
        try:
            # Neural network-based inpainting
            inpainted_img = neural_inpainting(image)
        except Exception as e:
            print(f"Error in inpainting process: {e}")
            # Fallback: return the original image
            inpainted_img = image

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_inpainted{ext}")
        if cv2 is not None:
            cv2.imwrite(output_path, inpainted_img)
        else:
            # Convert numpy array back to PIL Image and save
            enhanced_pil = Image.fromarray(inpainted_img)
            enhanced_pil.save(output_path)

        return output_path, None, None

    else:
        # Return original if task not recognized
        return img_path, None, None

@app.route('/')
def index():
    """Home page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload"""
    if request.method == 'POST':
        # Detailed debugging
        print("=== UPLOAD DEBUG INFO ===")
        print(f"Request method: {request.method}")
        print(f"Request files keys: {list(request.files.keys())}")
        print(f"Request form keys: {list(request.form.keys())}")
        print(f"Content-Type: {request.content_type}")
        print(f"Raw request data length: {len(request.get_data())}")
        print(f"Request data preview: {request.get_data()[:200]}...")  # First 200 chars

        # Check if file was submitted
        if 'file' not in request.files:
            print("ERROR: 'file' not in request.files")
            print(f"Available files: {list(request.files.keys())}")
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        # Debug: print filename info
        print(f"File object: {file}")
        print(f"File filename: '{file.filename}'")
        print(f"File content type: {file.content_type}")
        if file.filename:  # Only try to read if filename is not empty
            print(f"File size: {len(file.read()) if file else 0}")
            file.seek(0)  # Reset file pointer after reading for size check

        # Check if file was actually selected
        if file.filename == '':
            print("ERROR: file.filename is empty")
            flash('No file selected')
            return redirect(request.url)

        # Check if file is allowed and save it
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + "_" + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Redirect to task selection page
            return redirect(url_for('select_task', filename=unique_filename))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)

    return redirect(url_for('index'))

@app.route('/task/<filename>', methods=['GET', 'POST'])
def select_task(filename):
    """Select enhancement task for the uploaded image"""
    if request.method == 'POST':
        task_type = request.form.get('task')
        if task_type:
            try:
                # Apply selected enhancement
                output_path, psnr_val, ssim_val = enhance_image(
                    os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                    task_type
                )
                
                # Extract just the filename from the path
                output_filename = os.path.basename(output_path)
                
                return render_template('result.html', 
                                     original_image=filename,
                                     enhanced_image=output_filename,
                                     task=task_type,
                                     psnr=psnr_val,
                                     ssim=ssim_val)
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('upload_file'))
    
    return render_template('task_selection.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)