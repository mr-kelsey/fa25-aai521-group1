# src/colorize/model_utils.py
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# --- Globals for Caffe Model ---
_CAFFE_MODEL = None
PROTOTXT = "models/colorization_deploy_v2.prototxt"
MODEL = "models/colorization_release_v2.caffemodel"
POINTS = "models/pts_in_hull.npy"

def load_caffe_model():
    """Loads the Caffe colorization model and cluster points."""
    global _CAFFE_MODEL
    if _CAFFE_MODEL is None:
        print("Loading Caffe colorization model...")
        
        if not Path(PROTOTXT).exists() or not Path(MODEL).exists():
            raise FileNotFoundError(
                f"Caffe model files not found. Ensure '{PROTOTXT}' and '{MODEL}' are in the 'models/' directory."
            )

        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        
        pts = np.load(POINTS)
        
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        
        # Set the blobs for the layers.
        # This handles cases where 'blobs' can be a tuple in some OpenCV versions.
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        _CAFFE_MODEL = net
    return _CAFFE_MODEL

def load_colorize_model(model_id: str):
    """Loads a colorization model based on the given ID."""
    if "caffe" in model_id.lower():
        return load_caffe_model()
    else:
        raise ValueError(f"Unsupported model_id for colorization: {model_id}")

def run_colorization(model, image: Image.Image):
    """
    Runs colorization on a PIL image using the Caffe model.
    """
    img_np = np.array(image.convert("RGB"))
    
    scaled = img_np.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))
    
    ab = cv2.resize(ab, (img_np.shape[1], img_np.shape[0]))
    
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    colorized = (255 * colorized).astype("uint8")
    
    return Image.fromarray(colorized)