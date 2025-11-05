\*\*Notebooks Overview\*\*



| Notebook | Purpose |

|----------|---------|

| \*\*denoising.ipynb\*\* | Loads a pre-trained denoising model (e.g., DDPM), fine-tunes it on noisy image samples, and evaluates its ability to remove noise while preserving image details. Includes visual comparisons and metrics like PSNR and SSIM. |

| \*\*super\_resolution.ipynb\*\* | Implements a super-resolution pipeline using models like VQ-VAE-2 or ESRGAN. Enhances low-resolution images and compares output against high-resolution ground truth. Includes training logs and visual benchmarks. |

| \*\*colorization.ipynb\*\* | Uses a generative model to colorize grayscale images. Fine-tunes on paired grayscale/color datasets and evaluates color fidelity. Includes side-by-side comparisons and histogram analysis. |

| \*\*inpainting.ipynb\*\* | Restores missing or damaged regions in images using inpainting models (e.g., Masked Autoencoders or diffusion-based models). Demonstrates seamless patch recovery and edge blending. |

| \*\*integration.ipynb\*\* | Combines all four enhancement tasks into a unified interface. Allows users to upload images, select a task, and view results. Serves as the backend logic for the optional Flask or Streamlit app. |





