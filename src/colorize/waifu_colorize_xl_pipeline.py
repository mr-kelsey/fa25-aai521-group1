import torch
from diffusers import DiffusionPipeline

class WaifuColorizeXLPipeline:
    def __init__(self, model_id="ShinoharaHare/Waifu-Colorize-XL"):
        # Load the custom pipeline from model repo
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        self.pipe = self.pipe.to("cpu")

    def __call__(self, image, num_inference_steps=30, guidance_scale=7.5):
        """
        Call the model the way the original colorize_sdxl.py expects.
        """
        return self.pipe(
            image=image,
            control_image=image,
            prompt="anime style colorization",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
