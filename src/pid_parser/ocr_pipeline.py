"""
OCR pipeline for extracting metadata from engineering drawings.
"""

import os
import json
from typing import Dict, Any, List, Optional

from PIL import Image
from openai import OpenAI
import mlflow
from mlflow.entities import SpanType

from .image_utils import (
    pdf_page_to_image,
    make_preview,
    image_to_base64,
    save_image,
    rotate_image,
    crop_from_normalized,
    resize_for_ocr,
    get_pdf_page_count,
)
from .api_utils import extract_text_from_response, extract_json_from_response
from .prompts import ORIENTATION_PROMPT, LAYOUT_PROMPT, OCR_PROMPT


def direction_to_rotation(direction: str) -> int:
    """
    Map text direction to clockwise rotation needed to make text horizontal left-to-right.
    
    Args:
        direction: One of "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"
    
    Returns:
        Degrees clockwise to rotate (0, 90, 180, or 270)
    """
    mapping = {
        "left_to_right": 0,      # already correct
        "right_to_left": 180,    # upside-down horizontally
        "top_to_bottom": 90,     # need 90° CW to make L→R
        "bottom_to_top": 270,    # need 270° CW (or 90° CCW)
    }
    return mapping.get(direction, 0)


@mlflow.trace(name="detect_page_rotation", span_type=SpanType.CHAIN)
def detect_page_rotation(
    full_img: Image.Image,
    fm_client: OpenAI,
    model_name: str,
    force_rotation: Optional[int] = None,
) -> int:
    """
    Detect rotation needed for the full page image.
    
    Uses a simpler approach: ask LLM which direction text runs, then map to rotation.
    This is more reliable than asking LLM to compute rotation directly.
    
    Args:
        full_img: Full page PIL Image
        fm_client: OpenAI-compatible client for FMAPI
        model_name: Model name to use for rotation detection
        force_rotation: Override auto-detection (0, 90, 180, 270)
    
    Returns:
        Degrees clockwise to rotate (0, 90, 180, or 270)
    """
    if force_rotation is not None:
        print(f"  Using forced rotation: {force_rotation}°")
        return force_rotation
    
    # Create preview for rotation detection
    preview = make_preview(full_img, max_side=2048)
    img_b64 = image_to_base64(preview)
    
    print(f"  Preview size for rotation detection: {preview.size}")

    completion = fm_client.chat.completions.create(
        model=model_name,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": ORIENTATION_PROMPT},
                ],
            }
        ],
    )

    raw_text = extract_text_from_response(completion)
    print(f"  Orientation response: {raw_text.strip()}")
    
    # Parse direction
    json_str = extract_json_from_response(raw_text)
    try:
        result = json.loads(json_str)
        direction = result.get("direction", "left_to_right")
        rotation = direction_to_rotation(direction)
        print(f"  Direction: {direction} → Rotation: {rotation}°")
        return rotation
    except (json.JSONDecodeError, ValueError) as e:
        # Try to extract direction from raw text (handle various formats)
        text_lower = raw_text.lower().replace(" ", "_").replace("-", "_")
        
        # Check for direction patterns
        direction_patterns = [
            ("top_to_bottom", ["top_to_bottom", "top_bottom", "vertical", "downward"]),
            ("bottom_to_top", ["bottom_to_top", "bottom_top", "upward"]),
            ("right_to_left", ["right_to_left", "right_left"]),
            ("left_to_right", ["left_to_right", "left_right", "horizontal"]),
        ]
        
        for direction, patterns in direction_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    rotation = direction_to_rotation(direction)
                    print(f"  Parsed direction: {direction} → Rotation: {rotation}°")
                    return rotation
        
        print(f"  Warning: Could not parse direction ({e}), defaulting to 0")
        return 0


@mlflow.trace(name="detect_title_block_region", span_type=SpanType.CHAIN)
def detect_title_block_region(
    rotated_img: Image.Image,
    fm_client: OpenAI,
    model_name: str,
) -> Dict[str, float]:
    """
    Detect title block region on the image.
    
    Args:
        rotated_img: Rotated PIL Image
        fm_client: OpenAI-compatible client for FMAPI
        model_name: Model name to use for layout detection
    
    Returns:
        Dict with x_min, y_min, x_max, y_max (normalized coordinates)
    """
    preview = make_preview(rotated_img, max_side=3096)
    img_b64 = image_to_base64(preview)
    
    # Default fallback (bottom-right quadrant)
    default_region = {"label": "title_block", "x_min": 0.5, "y_min": 0.7, "x_max": 1.0, "y_max": 1.0}
    
    try:
        completion = fm_client.chat.completions.create(
            model=model_name,
            max_tokens=2048,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": LAYOUT_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
        )
        print(f"  completion response: {completion}")
        raw_text = extract_text_from_response(completion)
        
        # Handle different response formats
        if isinstance(raw_text, list):
            for b in raw_text:
                if hasattr(b, "type") and b.type == "text":
                    raw_text = b.text
                    break
                if isinstance(b, dict) and b.get("type") == "text":
                    raw_text = b["text"]
                    break

        if raw_text is None or raw_text == "":
            raise RuntimeError("Claude (layout) response contained no text block.")

        # Extract JSON from response (handles markdown code blocks)
        json_str = extract_json_from_response(raw_text)
        result = json.loads(json_str)
        regions = result.get("regions", [])
        
        if regions:
            region = regions[0]
            print(f"  Detected region: {region}")
            return region
        
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Raw response: {raw_text[:300] if 'raw_text' in locals() else 'N/A'}...")
    
    print(f"  Using default: {default_region}")
    return default_region


@mlflow.trace(name="extract_metadata_ocr", span_type=SpanType.LLM)
def extract_metadata_ocr(
    crop_img: Image.Image,
    fm_client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    """
    Extract metadata from cropped title block image.
    
    Args:
        crop_img: Cropped title block image (already in correct orientation)
        fm_client: OpenAI-compatible client for FMAPI
        model_name: Model name to use for OCR extraction
    
    Returns:
        Dict with extracted_data, revision_history, etc.
    """
    img_b64 = image_to_base64(crop_img)

    completion = fm_client.chat.completions.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.15,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
    )

    raw_text = extract_text_from_response(completion)
    print(f"  OCR response (first 500 chars): {raw_text[:500]}...")
    
    json_str = extract_json_from_response(raw_text)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        raise RuntimeError(f"Failed to parse OCR JSON: {e}")


@mlflow.trace(name="process_engineering_drawing", span_type=SpanType.CHAIN)
def process_engineering_drawing(
    pdf_path: str,
    fm_client: OpenAI,
    ocr_model: str,
    layout_model: str,
    page_index: int = 0,
    output_path: Optional[str] = None,
    dpi: int = 400,
    force_rotation: Optional[int] = None,
    min_crop_height: int = 2048,
) -> Dict[str, Any]:
    """
    Process an engineering drawing PDF to extract title block metadata.
    
    Pipeline:
    1. PDF → Full image
    2. Detect rotation on full page
    3. Rotate full image
    4. Detect title block region on rotated image
    5. Crop title block
    6. OCR the crop
    
    Args:
        pdf_path: Path to PDF file
        fm_client: OpenAI-compatible client for FMAPI
        ocr_model: Model name for OCR extraction
        layout_model: Model name for layout detection
        page_index: Page number (0-indexed)
        output_path: Where to save cropped image (optional)
        dpi: Resolution for PDF rendering
        force_rotation: Override rotation detection (0, 90, 180, 270)
        min_crop_height: Minimum height for OCR
    
    Returns:
        Dict with metadata, region info, rotation applied, etc.
    """
    print(f"Processing: {pdf_path}, page {page_index}")
    print("=" * 60)
    
    # Initialize image paths dict for logging
    image_paths = {
        "full_page_image": None,
        "rotated_image": None,
        "cropped_title_block": None,
    }
    
    # Determine base output path for intermediate images
    if output_path:
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
    else:
        base_dir = None
        base_name = None
    
    # Step 1: PDF to image
    print("\n[Step 1] Converting PDF to image...")
    full_img = pdf_page_to_image(pdf_path, page_index=page_index, dpi=dpi)
    print(f"  Full image size: {full_img.size}")
    
    # Save full page image if output directory provided
    if base_dir and base_name:
        full_page_path = os.path.join(base_dir, f"{base_name}_full.png")
        save_image(full_img, full_page_path)
        image_paths["full_page_image"] = full_page_path
        print(f"  Saved full page: {full_page_path}")
    
    # Step 2: Detect rotation on FULL PAGE
    print("\n[Step 2] Detecting page rotation...")
    rotation = detect_page_rotation(full_img, fm_client, ocr_model, force_rotation=force_rotation)
    print(f"  Rotation needed: {rotation}°")
    
    # Step 3: Rotate full image
    print("\n[Step 3] Applying rotation to full image...")
    if rotation != 0:
        rotated_img = rotate_image(full_img, rotation)
        print(f"  Rotated image size: {rotated_img.size}")
        
        # Save rotated image
        if base_dir and base_name:
            rotated_path = os.path.join(base_dir, f"{base_name}_rotated.png")
            save_image(rotated_img, rotated_path)
            image_paths["rotated_image"] = rotated_path
            print(f"  Saved rotated: {rotated_path}")
    else:
        rotated_img = full_img
        image_paths["rotated_image"] = image_paths["full_page_image"]  # Same as full
        print("  No rotation needed")
    
    # Step 4: Detect title block region on ROTATED image
    print("\n[Step 4] Detecting title block region...")
    region = detect_title_block_region(rotated_img, fm_client, layout_model)
    
    # Step 5: Crop from rotated image
    print("\n[Step 5] Cropping title block...")
    crop = crop_from_normalized(rotated_img, region)
    print(f"  Crop size: {crop.size}")
    
    # Upscale if needed
    crop_for_ocr = resize_for_ocr(crop, min_height=min_crop_height)
    if crop_for_ocr.size != crop.size:
        print(f"  Upscaled to: {crop_for_ocr.size}")
    
    # Save crop if output path provided
    if output_path:
        print(f"\n[Step 5b] Saving crop to: {output_path}")
        save_image(crop_for_ocr, output_path)
        image_paths["cropped_title_block"] = output_path
    
    # Step 6: OCR
    print("\n[Step 6] Extracting metadata via OCR...")
    metadata = extract_metadata_ocr(crop_for_ocr, fm_client, ocr_model)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    
    # Log image paths summary
    print("\nImage paths:")
    for key, path in image_paths.items():
        if path:
            print(f"  {key}: {path}")

    return {
        "rotation_applied": rotation,
        "region": region,
        "metadata": metadata,
        "output_path": output_path,
        "image_paths": image_paths,
        "original_size": full_img.size,
        "rotated_size": rotated_img.size,
        "crop_size": crop_for_ocr.size,
    }


def process_single_drawing_safe(
    pdf_path: str,
    fm_client: OpenAI,
    ocr_model: str,
    layout_model: str,
    page_index: int = 0,
    output_dir: Optional[str] = None,
    dpi: int = 400,
    force_rotation: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process a single drawing page with error handling.
    Returns a result dict that includes success/failure status.
    
    Args:
        pdf_path: Path to PDF file
        fm_client: OpenAI-compatible client for FMAPI
        ocr_model: Model name for OCR extraction
        layout_model: Model name for layout detection
        page_index: Page number (0-indexed)
        output_dir: Directory to save cropped images (optional)
        dpi: Resolution for PDF rendering
        force_rotation: Override rotation detection (0, 90, 180, 270)
    
    Returns:
        Dict with processing results and metadata
    """
    result = {
        "pdf_path": pdf_path,
        "pdf_filename": os.path.basename(pdf_path),
        "page_index": page_index,
        "total_pages": None,
        "status": "success",
        "error_message": None,
        "rotation_applied": None,
        "region": None,
        "contractor": None,
        "project_title": None,
        "drawing_title": None,
        "unit": None,
        "plant": None,
        "contr_proj_no": None,
        "agnoc_gas_dwg_no": None,
        "revision_history": None,
        "latest_rev": None,
        "latest_rev_date": None,
        "latest_rev_description": None,
        "revision_count": None,
        "extraction_errors": None,
        "confidence": None,
        "original_size": None,
        "crop_size": None,
    }
    
    try:
        # Generate output path if output_dir provided
        output_path = None
        if output_dir:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_page{page_index}_crop.png")
        
        # Process the drawing
        proc_result = process_engineering_drawing(
            pdf_path=pdf_path,
            fm_client=fm_client,
            ocr_model=ocr_model,
            layout_model=layout_model,
            page_index=page_index,
            output_path=output_path,
            dpi=dpi,
            force_rotation=force_rotation,
        )
        
        # Extract metadata fields
        result["rotation_applied"] = proc_result.get("rotation_applied")
        result["region"] = json.dumps(proc_result.get("region")) if proc_result.get("region") else None
        result["original_size"] = str(proc_result.get("original_size"))
        result["crop_size"] = str(proc_result.get("crop_size"))
        
        metadata = proc_result.get("metadata", {})
        extracted = metadata.get("extracted_data", {})
        
        result["contractor"] = extracted.get("contractor")
        result["project_title"] = extracted.get("project_title")
        result["drawing_title"] = extracted.get("drawing_title")
        result["unit"] = extracted.get("unit")
        result["plant"] = extracted.get("plant")
        result["contr_proj_no"] = extracted.get("contr_proj_no")
        result["agnoc_gas_dwg_no"] = extracted.get("agnoc_gas_dwg_no")
        
        # Store full revision history as JSON
        rev_history = metadata.get("revision_history", [])
        result["revision_history"] = json.dumps(rev_history)
        result["revision_count"] = len(rev_history)
        
        # Extract latest revision details (last in the list)
        if rev_history:
            latest = rev_history[-1]
            result["latest_rev"] = latest.get("rev")
            result["latest_rev_date"] = latest.get("date")
            result["latest_rev_description"] = latest.get("description")
        
        result["extraction_errors"] = json.dumps(metadata.get("extraction_errors", []))
        result["confidence"] = metadata.get("confidence")
        
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        print(f"  ERROR processing {pdf_path} page {page_index}: {e}")
    
    return result


def process_folder(
    input_folder: str,
    fm_client: OpenAI,
    ocr_model: str,
    layout_model: str,
    output_dir: Optional[str] = None,
    process_last_page: bool = True,
    dpi: int = 400,
    force_rotation: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a folder, extracting metadata from the LAST page of each PDF.
    
    Args:
        input_folder: Path to folder containing PDF files
        fm_client: OpenAI-compatible client for FMAPI
        ocr_model: Model name for OCR extraction
        layout_model: Model name for layout detection
        output_dir: Directory to save cropped images (optional)
        process_last_page: If True (default), process only the last page of each PDF
        dpi: Resolution for PDF rendering
        force_rotation: Override rotation detection for all drawings
    
    Returns:
        List of result dictionaries, one per processed PDF
    """
    results = []
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted([
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.lower().endswith('.pdf')
    ])
    
    print(f"Found {len(pdf_files)} PDF files in {input_folder}")
    print("=" * 80)
    
    for i, pdf_path in enumerate(pdf_files):
        pdf_filename = os.path.basename(pdf_path)
        print(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf_filename}")
        
        try:
            # Get page count
            page_count = get_pdf_page_count(pdf_path)
            print(f"  Total pages: {page_count}")
            
            # Process the last page (index = page_count - 1)
            last_page_index = page_count - 1
            print(f"  Processing last page (page {last_page_index + 1})")
            
            # Create a SEPARATE TRACE for each drawing - include full path in name
            with mlflow.start_span(name=f"ocr:{pdf_path}", span_type=SpanType.CHAIN) as span:
                # Set attributes for this trace
                span.set_attributes({
                    "pdf_path": pdf_path,
                    "pdf_filename": pdf_filename,
                    "page_index": last_page_index,
                    "total_pages": page_count,
                    "dpi": dpi,
                })
                
                result = process_single_drawing_safe(
                    pdf_path=pdf_path,
                    fm_client=fm_client,
                    ocr_model=ocr_model,
                    layout_model=layout_model,
                    page_index=last_page_index,
                    output_dir=output_dir,
                    dpi=dpi,
                    force_rotation=force_rotation,
                )
                
                # Store total page count in result
                result["total_pages"] = page_count
                
                # Log result status and image paths to span
                span_attrs = {
                    "status": result.get("status", "unknown"),
                    "confidence": result.get("confidence", "unknown"),
                }
                
                # Log all image paths for reference
                image_paths = result.get("image_paths", {})
                if image_paths:
                    for img_type, img_path in image_paths.items():
                        if img_path:
                            span_attrs[f"image.{img_type}"] = img_path
                
                # Log crop output path
                if result.get("output_path"):
                    span_attrs["image.crop_output"] = result.get("output_path")
                
                span.set_attributes(span_attrs)
                
            results.append(result)
                
        except Exception as e:
            print(f"  FATAL ERROR with {pdf_path}: {e}")
            results.append({
                "pdf_path": pdf_path,
                "pdf_filename": pdf_filename,
                "page_index": None,
                "total_pages": None,
                "status": "fatal_error",
                "error_message": str(e),
            })
    
    print("\n" + "=" * 80)
    print(f"Processing complete! Processed {len(results)} pages from {len(pdf_files)} PDFs")
    
    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = len(results) - success_count
    print(f"  Success: {success_count}, Errors: {error_count}")
    
    return results

