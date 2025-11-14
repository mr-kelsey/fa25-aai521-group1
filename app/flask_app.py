from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import uuid
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key_for_csrf_protection'

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
    # Convert to float for calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert BGR to RGB for scikit-image compatibility
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    return ssim(img1_rgb, img2_rgb, multichannel=True, channel_axis=-1)

def simple_denoise(image):
    """Simple denoising using Gaussian blur"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def denoise_image(noisy_img):
    """Apply denoising to the noisy image"""
    denoised_img = simple_denoise(noisy_img)

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(noisy_img, denoised_img)
    ssim_value = calculate_ssim(noisy_img, denoised_img)

    return denoised_img, psnr_value, ssim_value

def enhance_image(img_path, task_type):
    """Apply selected enhancement task to the image"""
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Could not load image")

    if task_type == 'denoising':
        # Add noise to the image for demonstration purposes
        noisy_img = add_noise_to_image(image)

        # Apply denoising
        enhanced_img, psnr_val, ssim_val = denoise_image(noisy_img)

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_denoised{ext}")
        cv2.imwrite(output_path, enhanced_img)

        return output_path, psnr_val, ssim_val

    elif task_type == 'super_resolution':
        # Simple super resolution by resizing (for demonstration)
        height, width = image.shape[:2]
        # Upscale by 2x
        upscaled_img = cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_super_res{ext}")
        cv2.imwrite(output_path, upscaled_img)

        return output_path, None, None

    elif task_type == 'colorization':
        # Simple colorization by converting grayscale to color (if grayscale)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        colorized_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_colorized{ext}")
        cv2.imwrite(output_path, colorized_img)

        return output_path, None, None

    elif task_type == 'inpainting':
        # Simple inpainting (for demonstration)
        # Create a mask with some region removed
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = image.shape[:2]
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

        # Simply fill the masked region with average of the surroundings
        result = image.copy()
        mean_color = np.mean(image[mask==0], axis=0)  # Average of non-masked pixels
        result[mask==255] = mean_color  # Fill masked region with average

        # Save enhanced image
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{name}_inpainted{ext}")
        cv2.imwrite(output_path, result)

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
        # Check if file was submitted
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
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
    
    return render_template('upload.html')

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