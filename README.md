# Group 1 final project for AAI-521

This project is a part of the AAI-521 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: [Active, Completed]

## Overview

This project implements an AI-powered image enhancement tool that includes four main computer vision tasks:
1. **Denoising** - Removing noise from images
2. **Super Resolution** - Enhancing low-resolution images
3. **Colorization** - Adding color to grayscale images
4. **Inpainting** - Restoring missing or damaged regions in images

The system leverages state-of-the-art generative models from Hugging Face, including diffusion models and specialized architectures for each enhancement task.

## Installation

To run this project on your machine, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/mr-kelsey/fa25-aai521-group1.git
cd fa25-aai521-group1
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. If requirements.txt is empty or missing, install the core dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install flask opencv-python pillow scikit-image matplotlib
pip install datasets jupyter
```

## Project Intro/Objective

The main purpose of this project is to create a comprehensive AI-powered image enhancement suite that demonstrates state-of-the-art techniques in computer vision. The goal is to develop a web application that allows users to upload images and apply advanced enhancement techniques using transformer-based models and diffusion processes.

Our application addresses the growing need for high-quality image processing in various domains including photography, medical imaging, satellite imagery, and historical photo restoration. The system leverages generative models to restore, enhance, and improve the quality of digital images through automated AI-driven processes.

## Partner(s)/Contributor(s)

* Johnathan Kelsey
* [https://github.com/mr-kelsey](https://github.com/mr-kelsey)
* Gregory Bauer
* [Gregory_github_account]
* Atul Aneja
* [Atul_github_account]

## Methods Used
* Inferential Statistics
* NLP
* Computer Vision
* Machine Learning
* Deep Learning
* Ethics for AI
* Data Visualization
* Data Manipulation
* Generative Modeling
* Diffusion Models

## Technologies
* Python
* PyTorch
* Transformers (Hugging Face)
* Diffusers (Hugging Face)
* Flask
* OpenCV
* Scikit-image
* NumPy
* Matplotlib
* Jupyter Notebooks
* CUDA (for GPU acceleration)

## Project Description

This project consists of four interconnected image enhancement modules, each leveraging specialized deep learning models:

### Computer Vision Tasks Implemented:
- **Denoising**: Uses diffusion models to remove various types of noise while preserving important image details
- **Super Resolution**: Employs convolutional and transformer-based models to enhance image resolution and quality
- **Colorization**: Applies neural networks trained on color mapping to add realistic colors to grayscale images
- **Inpainting**: Implements inpainting models to reconstruct missing or damaged portions of images

### Dataset and Architecture:
The project uses a comprehensive dataset that includes ground truth high-quality images alongside their artificially degraded versions (noisy, low-resolution, grayscale, corrupted). The architecture combines Hugging Face's pre-trained models with custom implementations to handle the four enhancement tasks.

### Technical Implementation:
- Jupyter notebooks for experimentation and model prototyping
- Flask web application for user interaction
- Modular design allowing easy addition of new enhancement tasks
- Comprehensive evaluation with PSNR and SSIM metrics
- GPU acceleration support for faster processing

### Challenges Addressed:
- Balancing enhancement quality with computational efficiency
- Maintaining semantic consistency during restoration
- Handling various types of image degradations
- Providing intuitive user interface for non-technical users

## Usage

### Running the Web Application
1. Navigate to the project directory
2. Run the Flask application:
```bash
cd app
python flask_app.py
```
3. Open your browser and navigate to `http://localhost:5000`
4. Upload an image and select the enhancement task you want to apply

### Using Jupyter Notebooks
1. Navigate to the notebooks directory
2. Launch Jupyter:
```bash
cd notebooks
jupyter notebook
```
3. Open the relevant notebook (`denoising.ipynb`, `super_resolution.ipynb`, etc.) to experiment with individual enhancement techniques

## License
GNU GENERAL PUBLIC LICENSE Version 3

## Acknowledgments
We thank our professor Dr. Sardari for their guidance and technical support throughout this project. Special thanks to the Hugging Face team for providing accessible state-of-the-art models, and the open-source community for the various libraries that made this project possible.