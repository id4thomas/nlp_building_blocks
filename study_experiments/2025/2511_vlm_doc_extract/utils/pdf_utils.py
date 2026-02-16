import os

from typing import Literal
from pdf2image import convert_from_bytes

def convert_pdf_to_png(
    source_fpath: str,
    result_dir: str,
    fmt: Literal["png", "jpeg", "jpg"] = "jpeg"
) -> int:
    with open(source_fpath, 'rb') as pdf:
        images = convert_from_bytes(pdf.read(), fmt=fmt)
        
    for image_i, image in enumerate(images):
        image.save(os.path.join(result_dir, f"page{image_i+1}.{fmt}"))
    
    return len(images)