import base64
import io
from typing import List, Dict, Any

from PIL import Image

def unnormalize_bbox(bbox_dict, width, height):
    x1 = bbox_dict["x1"]
    y1 = bbox_dict["y1"]
    x2 = bbox_dict["x2"]
    y2 = bbox_dict["y2"]
    
    bx1 = int(x1 / 1000 * width)
    by1 = int(y1 / 1000 * height)
    bx2 = int(x2 / 1000 * width)
    by2 = int(y2 / 1000 * height)
    if bx1 > bx2:
        bx1, bx2 = bx2, bx1
    if by1 > by2:
        by1, by2 = by2, by1
    return bx1, by1, bx2, by2

def load_image(image_path):
    return Image.open(image_path)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return encoded_image

def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """
    Converts a PIL.Image.Image object to a base64 encoded string.

    Args:
        pil_image: The PIL.Image.Image object to convert.
        format: The image format to use (e.g., "PNG", "JPEG").

    Returns:
        A base64 encoded string.
    """
    # Create an in-memory buffer
    buffered = io.BytesIO()
    
    # Save the image to the buffer in the specified format
    # Handle RGBA to RGB conversion for formats like JPEG
    if format == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
        
    pil_image.save(buffered, format=format)
    
    # Get the byte value from the buffer
    img_bytes = buffered.getvalue()
    
    # Encode the bytes to base64 and decode to a UTF-8 string
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    
    return base64_str

def crop_image(
    image: Image.Image,
    bbox: Dict[str, int]
):
    width, height = image.size
    bx1, by1, bx2, by2 = unnormalize_bbox(bbox, width=width, height=height)
    crop_box = (bx1, by1, bx2, by2)
    cropped_img = image.crop(crop_box)
    return cropped_img