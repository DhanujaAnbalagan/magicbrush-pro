import base64
import io
import cv2
import numpy as np
from PIL import Image

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decodes a base64 string to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def encode_image_base64(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def preprocess_image_and_mask(init_image: Image.Image, init_mask: Image.Image):
    """
    Convert to RGB, resize to 512x512, and refine the mask.
    """
    # Resize to 512x512
    init_image = init_image.resize((512, 512), Image.Resampling.LANCZOS)
    init_mask = init_mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    init_image = init_image.convert("RGB")
    init_mask = init_mask.convert("L")  # Grayscale mask

    # Refine mask using OpenCV (dilation and feathering/blur)
    mask_np = np.array(init_mask)
    
    # Thresholding to ensure binary
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # Dilation to slightly expand mask
    kernel = np.ones((5, 5), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    # Gaussian blur for feathering
    mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
    
    refined_mask = Image.fromarray(mask_np)

    return init_image, refined_mask
