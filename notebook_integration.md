fa25-aai521-group1/

│

├── helper/                          # teammate modules (remain intact)

│   ├── completion.py                 # Completion class (patch extraction, stitching, visualization)

│   └── utils.py                      # augmentations (damage, noise, scale, grayscale)

│

├── src/

│   └── superres/                     # new modular pipeline package

│       ├── \_\_init\_\_.py               # package initializer, exposes public API

│       ├── model\_utils.py            # load LDM model, run inference

│       ├── single\_image.py           # main entry point: enhance\_image()

│       └── viz\_utils.py              # optional visualization helpers

│

└── superres\_pipeline.py              # CLI entry point for running pipeline



