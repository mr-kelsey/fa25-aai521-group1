# Qwen Context File for AAI-521 Group 1 Final Project

## Project Overview

This is an AI-powered image enhancement tool developed by Group 1 for the AAI-521 course at the University of San Diego. The project implements four main computer vision tasks:

1. **Denoising** - Removing noise from images using diffusion models
2. **Super Resolution** - Enhancing low-resolution images using transformer-based models
3. **Colorization** - Adding color to grayscale images using neural networks
4. **Inpainting** - Restoring missing or damaged regions in images using inpainting models

The system leverages state-of-the-art generative models from Hugging Face, including diffusion models and specialized architectures for each enhancement task.

## Directory Structure

```
├── .gitignore
├── colab_test.txt
├── Group_Schedule_3P.md
├── LICENSE
├── README.md
├── requirements.txt
├── SUMMARY.md
├── technical_report.md
├── .git/
├── .idea/
├── .venv/
├── app/
│   ├── flask_app.py
│   └── templates/
├── data/
│   ├── damaged/
│   ├── grey/
│   ├── noisy/
│   ├── scaled/
│   └── truth/
├── notebooks/
│   ├── helper/
│   ├── superres/
│   ├── colorization.ipynb
│   ├── denoising.ipynb
│   ├── inpainting.ipynb
│   ├── integration.ipynb
│   ├── README.md
│   └── super_resolution.ipynb
└── outputs/
```

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: PyTorch, Transformers (Hugging Face), Diffusers (Hugging Face), Accelerate
- **Computer Vision**: OpenCV, Scikit-image
- **Image Processing**: Pillow, NumPy, Matplotlib
- **Development**: Jupyter Notebooks, CUDA (for GPU acceleration)

## Key Files and Components

### Web Application (`app/flask_app.py`)
- Main Flask application implementing the web interface for image enhancement
- Handles file uploads, task selection, and result display
- Implements all four enhancement tasks with basic metrics (PSNR, SSIM)
- Includes templates for user interaction

### Notebooks (`notebooks/`)
- `colorization.ipynb`: Implements colorization functionality
- `denoising.ipynb`: Implements denoising functionality
- `inpainting.ipynb`: Implements inpainting functionality
- `super_resolution.ipynb`: Implements super-resolution functionality
- `integration.ipynb`: Integrates all four tasks
- `helper/utils.py`: Helper functions for image processing

### Data (`data/`)
- `damaged/`: Contains images with damaged regions for inpainting
- `grey/`: Contains grayscale images for colorization
- `noisy/`: Contains noisy images for denoising
- `scaled/`: Contains low-resolution images for super-resolution
- `truth/`: Contains ground truth high-quality images

### Outputs (`outputs/`)
- Stores enhanced images and evaluation metrics after processing

## Building and Running

### Installation
1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Install additional dependencies if needed:
```bash
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install flask opencv-python pillow scikit-image matplotlib
pip install datasets jupyter
```

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

## Project Architecture

The project follows a modular architecture with:
- Separate implementations for each enhancement task
- A Flask web application serving as a unified interface
- Jupyter notebooks for experimentation and model prototyping
- Comprehensive evaluation with PSNR and SSIM metrics
- GPU acceleration support for faster processing

## Development Conventions

- Code is primarily written in Python
- Jupyter notebooks are used for experimentation
- Modular design allows easy addition of new enhancement tasks
- Evaluation metrics (PSNR, SSIM) are computed for quality assessment
- The project uses Hugging Face models for state-of-the-art results

## Key Features

- Web interface for easy user interaction
- Support for multiple image enhancement tasks
- Quality metrics computation
- GPU acceleration for faster processing
- Modular design allowing for extensibility
- Comprehensive documentation and technical report

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.

## Local Working Branch Workflow

This branch (`joeldiev/workingbranch`) serves as a local working branch with special considerations:

1. **Local-Only Files**:
   - `QWEN.md` (this file) should remain only in this local branch
   - `TODO.md` should remain only in this local branch
   - These files contain local development context and task tracking

2. **Branch Management**:
   - When making code changes, update from main and create new feature branches
   - Apply changes to the feature branch, not this local working branch
   - Merge feature branches back to main as appropriate
   - Keep this local branch for ongoing development context

3. **File Handling**:
   - If switching from this branch to work on main, stash `QWEN.md` and `TODO.md`
   - When returning to this local branch, restore stashed files
   - This prevents conflicts and keeps local context files isolated

4. **Synchronization**:
   - Regularly update this local branch with changes from main
   - Use `git stash` and `git stash pop` as needed when switching contexts
   - This branch should not be pushed to remote repository

5. **Branch Creation Rule**:
   - Never merge unrelated files (such as base documentation files) directly into feature branches
   - Always create new feature branches from main for specific feature work
   - Develop features on dedicated branches before merging to main
   - This prevents file contamination and keeps feature branches focused