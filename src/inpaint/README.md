# Inpainting Pipeline (AAI-521 Final Project)

This project provides a modular pipeline for single-image **inpainting** using:

- Stable Diffusion Inpainting (Hugging Face Diffusers)
- Patch-based Completion (team contribution)

The pipeline compares the original image to the inpainted image, saving both for analysis and visualization.

---

## Installation

Clone the repository and install dependencies:

```bash
py -m pip install -r requirements.txt
```

Dependencies are pinned for reproducibility:
- Torch / TorchVision / Torchaudio (PyTorch ≥ 2.6 required)
- Diffusers / Transformers / Accelerate
- OpenCV / Pillow / Matplotlib
- NumPy / Pandas
- Scikit-Image / Scikit-Learn

---

## Usage

Run the pipeline on a single damaged image:

```bash
py inpaint_pipeline.py --image data/damaged/0006_d.png --method sd --save-to outputs/0006_inpaint.png
```

### CLI Options

- `--image` : Path to input image (required)  
- `--method` : Enhancement method (`sd` for Stable Diffusion Inpainting, `completion` for patch‑based Completion)  
- `--model-id` : Hugging Face model ID (default: `runwayml/stable-diffusion-inpainting`)  
- `--save-to` : Base path for saving outputs (default: `outputs/result.png`)  
- `--visualize` : Show before/after comparison in a matplotlib window  
- `--save-only` : Save outputs without visualization  

---

## Output

Each run produces two files:

- `<basename>_input.png` : The original damaged image  
- `<basename>_inpainted.png` : The inpainted image  

Example (Stable Diffusion Inpainting):

```bash
py inpaint_pipeline.py --image data/damaged/0006_d.png --method sd --save-to outputs/0006_inpaint.png --save-only
```

Produces:

```
outputs/0006_inpaint_input.png
outputs/0006_inpainted.png
```

Example (Completion):

```bash
py inpaint_pipeline.py --image data/damaged/0006_d.png --method completion --save-to outputs/0006_completion.png
```

Produces:

```
outputs/0006_completion_input.png
outputs/0006_completion_inpainted.png
```

---

## Model Notes

- The **old Hugging Face inpainting model** (`stabilityai/stable-diffusion-inpainting`) listed in `HuggingFace_models.md` is **no longer available**. Attempts to fetch it result in a 404 error.  
- The recommended replacement is **`runwayml/stable-diffusion-inpainting`**, which remains publicly accessible and works with the Diffusers library.  
- This model is specifically trained for **inpainting tasks**: it takes an input image and a binary mask, and fills in the masked regions with contextually appropriate content.  
- In our pipeline, masks are generated automatically by thresholding bright specks (damage) in the input image. This allows the model to restore corrupted regions without altering the rest of the image.  
- The model supports optional text prompts (`--prompt`) to guide the inpainting process (e.g., “restore missing text” or “fill with foliage”), though our default pipeline runs without prompts for unbiased restoration.

---

## Quick Comparison

| Feature                | Stable Diffusion Inpainting (`sd`)   | Completion Method (`completion`)         |
|-------------------------|--------------------------------------|------------------------------------------|
| Model ID                | `runwayml/stable-diffusion-inpainting` | Patch-based Completion class             |
| Input size              | Arbitrary (resized internally)       | Arbitrary, padded to patch size          |
| Enhancement approach    | Hugging Face Stable Diffusion model  | Patch-based stitching via Completion     |
| Visualization           | Controlled by `--visualize` flag     | Always shows matplotlib window internally|
| Output files            | `<basename>_input.png`, `<basename>_inpainted.png` | Same naming convention                   |
| CLI flexibility         | Supports `--model-id` and `--prompt` | Supports patch size parameter            |

---

## Pipeline Flow Diagram

### Stable Diffusion Inpainting
```
Input Image (damaged)
        ↓
Mask Generation (threshold bright specks)
        ↓
Stable Diffusion Inpainting (Hugging Face)
        ↓
Inpainted Image
        ↓
Save: *_input.png and *_inpainted.png
```

### Completion Method
```
Input Image (any size)
        ↓
Patch Extraction (based on --completion-patch)
        ↓
Patch Processing (pipeline function, e.g., upscaling)
        ↓
Stitching into full image
        ↓
Visualization (always shown by Completion class)
        ↓
Save: *_input.png and *_inpainted.png
```

---

## Project Structure

```
fa25-aai521-group1/
├── inpaint_pipeline.py        # CLI entry point
├── src/
│   └── inpaint/
│       ├── single_image.py    # Core inpainting logic
│       ├── model_utils.py     # Model loading utilities
│       └── viz_utils.py       # Visualization helpers
├── notebooks/
│   └── helper/
│       ├── __init__.py
│       ├── completion.py      # Patch-based Completion class
│       └── utils.py           # Helper functions (damage, noise, scaling, etc.)
└── outputs/                   # Saved results
```

---

## Team Contributions

- Stable Diffusion Inpainting: Integrated via Hugging Face Diffusers  
- Completion Method: Full Completion class used for patch-based enhancement  
- Pipeline Integration: Unified CLI with flexible options for saving and visualization  
- Documentation: Clear README and reproducible requirements for onboarding  

---

## Notes

- Ensure internet access for Hugging Face model downloads (first run may take longer).  
- Use `--save-only` for automated runs (no pop-up windows).  
- When using `--method completion`, the full Completion class is invoked. By design, it performs patch extraction, stitching, and visualization internally.  
- **Important:** This pipeline requires **PyTorch ≥ 2.6** due to Hugging Face security restrictions (CVE‑2025‑32434). Please upgrade your torch installation before running.  
- Outputs are reproducible across environments thanks to pinned dependencies.  

---

## License

This project is for academic use in AAI-521 Applied Computer Vision for AI.  
Please respect team contributions and cite appropriately in derivative work.
