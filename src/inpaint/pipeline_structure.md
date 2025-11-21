fa25-aai521-group1/
├── inpaint\_pipeline.py          # CLI entry point
├── src/
│   └── inpaint/
│       ├── **init**.py          # Package initializer
│       ├── single\_image.py      # Core inpainting logic
│       ├── model\_utils.py       # Model loading + inference
│       └── viz\_utils.py         # Visualization helpers

&nbsp;	└── requirements.txt    # Minimal dependencies for inpaint module
├── notebooks/
│   └── helper/
│       ├── **init**.py
│       ├── completion.py           # Mask generation utilities
│       └── utils.py             # Helper functions (noise, damage, etc.)
└── outputs/                     # Saved results

