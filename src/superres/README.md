---

```markdown
# Super-Resolution Pipeline (AAI-521 Final Project)

This project provides a modular pipeline for single-image enhancement using:

- Latent Diffusion Models (LDM) for super-resolution
- Patch-based Completion (team contribution)

The pipeline compares the original image to the enhanced image, saving both for analysis and visualization.

---

## Installation

Clone the repository and install dependencies:

```bash
py -m pip install -r requirements.txt
```

Dependencies are pinned for reproducibility:
- Torch / TorchVision (matched pair)
- Diffusers / Transformers / Accelerate
- OpenCV / Pillow / Matplotlib
- NumPy / Pandas
- Scikit-Image / Scikit-Learn

---

## Usage

Run the pipeline on a single image:

```bash
py superres_pipeline.py --image data/scaled/0006_x4.png --method ldm --save-to outputs/0006_ldm.png
```

### CLI Options

- `--image` : Path to input image (required)
- `--method` : Enhancement method (`ldm` or `completion`)
- `--completion-patch` : Patch size for completion method (default: `100 100`)
- `--model-id` : Hugging Face model ID for LDM (default: `CompVis/ldm-super-resolution-4x-openimages`)
- `--save-to` : Base path for saving outputs (default: `outputs/result.png`)
- `--visualize` : Show before/after comparison in a matplotlib window
- `--save-only` : Save outputs without visualization

---

## Output

Each run produces two files:

- `<basename>_input.png` : The original image (resized if needed)
- `<basename>_enhanced.png` : The enhanced image

Example (LDM):

```bash
py superres_pipeline.py --image data/scaled/0006_x4.png --method ldm --save-to outputs/0006_ldm.png --save-only
```

Produces:

```
outputs/0006_ldm_input.png
outputs/0006_ldm_enhanced.png
```

Example (Completion):

```bash
py superres_pipeline.py --image data/scaled/0006_x16.png --method completion --completion-patch 100 100 --save-to outputs/0006_completion.png
```

Produces:

```
outputs/0006_completion_input.png
outputs/0006_completion_enhanced.png
```

Note: The Completion class performs patch extraction, stitching, and visualization internally. A matplotlib window will appear by design when using this method.

---

## Quick Comparison

| Feature                | LDM Method                          | Completion Method                          |
|-------------------------|-------------------------------------|--------------------------------------------|
| Input size              | 128×128 (resized automatically)     | Arbitrary, padded to patch size             |
| Enhancement approach    | Hugging Face Latent Diffusion Model | Patch-based stitching via Completion class  |
| Visualization           | Controlled by `--visualize` flag    | Always shows matplotlib window internally   |
| Output files            | `<basename>_input.png`, `<basename>_enhanced.png` | Same naming convention                      |
| CLI flexibility         | Supports `--model-id` for different pretrained models | Supports `--completion-patch` for patch size |

---

## Pipeline Flow Diagram

### LDM Method
```
Input Image (resized to 128x128)
        ↓
Latent Diffusion Model (Hugging Face)
        ↓
Enhanced Image (super-resolved)
        ↓
Save: *_input.png and *_enhanced.png
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
Save: *_input.png and *_enhanced.png
```

---

## Project Structure

```
fa25-aai521-group1/
├── superres_pipeline.py        # CLI entry point
├── src/
│   └── superres/
│       ├── single_image.py     # Core enhancement logic
│       ├── model_utils.py      # Model loading utilities
│       └── viz_utils.py        # Visualization helpers
├── notebooks/
│   └── helper/
│       ├── __init__.py
│       ├── completion.py       # Full Completion class (patching, stitching, visualization)
│       └── utils.py            # Helper functions (noise, damage, etc.)
└── outputs/                    # Saved results
```

---

## Team Contributions

- LDM Super-Resolution: Integrated via Hugging Face Diffusers
- Completion Method: Full Completion class used for patch-based enhancement
- Pipeline Integration: Unified CLI with flexible options for saving and visualization
- Documentation: Clear README and reproducible requirements for onboarding

---

## Notes

- Ensure internet access for Hugging Face model downloads (first run may take longer).
- Use `--save-only` for automated runs (no pop-up windows).
- When using `--method completion`, the full Completion class is invoked. By design, it performs patch extraction, stitching, and visualization internally.
- Outputs are reproducible across environments thanks to pinned dependencies.

---

## License

This project is for academic use in AAI-521 Applied Computer Vision for AI.  
Please respect team contributions and cite appropriately in derivative work.
```

---