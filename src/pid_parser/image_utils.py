"""
Image processing utilities for PDF to image conversion and manipulation.
"""

import os
import io
import base64
from typing import Dict

import fitz  # PyMuPDF
from PIL import Image


def pdf_page_to_image(pdf_path: str, page_index: int = 0, dpi: int = 600) -> Image.Image:
    """
    Convert a single PDF page to a high-res PIL image using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        page_index: Page number (0-indexed)
        dpi: Resolution for rendering
    
    Returns:
        PIL Image object
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()
    return img


def make_preview(img: Image.Image, max_side: int = 3096) -> Image.Image:
    """
    Downscale image to max_side for sending to Claude (8k pixel limit).
    
    Args:
        img: PIL Image to downscale
        max_side: Maximum dimension for the longest side
    
    Returns:
        Resized PIL Image
    """
    w, h = img.size
    scale = min(max_side / float(max(w, h)), 1.0)
    if scale >= 1.0:
        return img.copy()
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(img: Image.Image) -> str:
    """
    Convert PIL image to base64 string.
    
    Args:
        img: PIL Image to convert
    
    Returns:
        Base64 encoded string
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, path: str) -> None:
    """
    Save image to path, creating directories if needed.
    
    Args:
        img: PIL Image to save
        path: Output file path
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    img.save(path, format="PNG")


def rotate_image(img: Image.Image, rotation_degrees: int) -> Image.Image:
    """
    Rotate image clockwise by specified degrees.
    
    Args:
        img: PIL Image to rotate
        rotation_degrees: Degrees to rotate clockwise (0, 90, 180, 270)
    
    Returns:
        Rotated PIL Image
    """
    if rotation_degrees == 0:
        return img
    # PIL rotates counter-clockwise, so negate for clockwise
    return img.rotate(-rotation_degrees, expand=True, resample=Image.BICUBIC)


def crop_from_normalized(img: Image.Image, box: Dict[str, float]) -> Image.Image:
    """
    Crop using normalized coords [0,1] on the image.
    
    Args:
        img: PIL Image to crop from
        box: Dict with x_min, y_min, x_max, y_max (normalized 0-1)
    
    Returns:
        Cropped PIL Image
    """
    w, h = img.size

    x_min = max(0.0, min(1.0, float(box["x_min"])))
    y_min = max(0.0, min(1.0, float(box["y_min"])))
    x_max = max(0.0, min(1.0, float(box["x_max"])))
    y_max = max(0.0, min(1.0, float(box["y_max"])))

    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))

    left = int(x_min * w)
    top = int(y_min * h)
    right = int(x_max * w)
    bottom = int(y_max * h)

    return img.crop((left, top, right, bottom))


def resize_for_ocr(crop: Image.Image, min_height: int = 1200, max_dimension: int = 7500) -> Image.Image:
    """
    Resize crop: upscale if too small, downscale if exceeds max dimension (8k limit).
    
    Args:
        crop: PIL Image to resize
        min_height: Minimum height for OCR quality
        max_dimension: Maximum dimension to avoid API limits
    
    Returns:
        Resized PIL Image
    """
    w, h = crop.size
    
    # First check if we need to downscale (max dimension limit)
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"  Downscaling from {w}x{h} to {new_w}x{new_h} (max dimension limit)")
        return crop.resize((new_w, new_h), Image.LANCZOS)
    
    # Then check if we need to upscale (min height)
    if h < min_height:
        scale = min_height / float(h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Make sure upscaling doesn't exceed max dimension
        if max(new_w, new_h) > max_dimension:
            scale = max_dimension / float(max(new_w, new_h))
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
        return crop.resize((new_w, new_h), Image.BICUBIC)

    return crop


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Number of pages
    """
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count

