"""
Prompt templates for vision model interactions.
"""

ORIENTATION_PROMPT = """Look at the TITLE BLOCK text (company name, project title, drawing number). Which way does the text run on screen?

{"direction": "left_to_right"}
{"direction": "right_to_left"}
{"direction": "top_to_bottom"}
{"direction": "bottom_to_top"}

Reply with ONLY the matching JSON. No explanation."""


LAYOUT_PROMPT = """
You are locating metadata on engineering drawings (P&ID, technical drawings, etc).

You are given a SINGLE PAGE IMAGE of a drawing. This image may be DOWNSCALED from the original.
Your job is to identify ONE rectangular region that contains BOTH:
1. The TITLE BLOCK (company, project, drawing number, scale, sheet, etc.)
2. The REVISION TABLE / REVISION HISTORY (revision numbers, dates, descriptions, approvals)
3. It is usuallly located at the bottom right corner.

If the revision table is embedded inside the title block, treat them together as one region.

COORDINATE SYSTEM (CRITICAL)
- You ONLY see the downscaled image.
- All coordinates MUST be NORMALIZED floats relative to THIS image:
  - 0.0 = minimum x or y of the image.
  - 1.0 = maximum x or y of the image.
- (x_min, y_min) is the TOP-LEFT corner of the box. typical value ranges from 0.5-.9
- (x_max, y_max) is the BOTTOM-RIGHT corner of the box.typical value ranges from 0.9-0.99
- All coordinates must be floats between 0.0 and 1.0 (e.g. 0.12, 0.87).
- NEVER output pixel values (such as 1047 or 490).
- You MUST ensure: x_min < x_max and y_min < y_max.
- The box must have a non-zero size in both width and height (DO NOT use identical min and max values).

WHAT TO INCLUDE
- The bounding box MUST encompass BOTH the title block AND the revision table.
- Include a  margin so that no relevant metadata is cut off.
- Avoid including large areas of unrelated drawing geometry.
- If you are uncertain, choose your single best estimate for this combined region.

THINKING
- First, think through the location of the title block and revision table.
- Then, decide on final coordinates.
- Do NOT write out your reasoning or any intermediate attempts; only output the final JSON.

OUTPUT FORMAT (STRICT)
Return ONLY valid JSON in this EXACT form, with your coordinates filled in:

{
  "regions": [
    {
      "label": "title_block_and_revisions",
      "x_min": 0.0,
      "y_min": 0.0,
      "x_max": 1.0,
      "y_max": 1.0
    }
  ]
}

OUTPUT CONSTRAINTS (VERY IMPORTANT)
- Output EXACTLY ONE JSON object.
- Do NOT wrap the JSON in ``` or any other code fences.
- Do NOT include any text before or after the JSON.
- Do NOT include explanations, apologies, or multiple versions.
- Do NOT say things like "let me reconsider" or "updated answer". Only provide the final JSON.
"""


OCR_PROMPT = """
Extract metadata from this engineering drawing title block.

RULES:
1. Extract ONLY text that is CLEARLY VISIBLE
2. Do NOT guess or hallucinate values
3. Use null for fields not found
4. The image is already in correct orientation - read it as displayed

Extract these fields:
- contractor: Company/contractor name
- project_title: Project title
- drawing_title: Drawing title
- unit: Unit number
- plant: Plant name  
- contr_proj_no: Contractor project number
- agnoc_gas_dwg_no: ADNOC Gas Drawing number

Also extract ALL visible revisions from the revision table.

Return ONLY this JSON:
{
  "extracted_data": {
    "contractor": null,
    "project_title": null,
    "drawing_title": null,
    "unit": null,
    "plant": null,
    "contr_proj_no": null,
    "agnoc_gas_dwg_no": null
  },
  "revision_history": [
    {"rev": "A", "date": "11.10.2024", "description": "ISSUED FOR REVIEW (IFR)", "drawn": "XX", "chkd": "YY", "apprd": "ZZ"}
  ],
  "extraction_errors": [],
  "confidence": "high"
}
"""


JUDGE_PROMPT = """You are an expert evaluator for OCR extraction from engineering drawings.

Compare the PREDICTED extraction against the GROUND TRUTH and assess the quality.

## Record Information
- **PDF File**: {pdf_filename}

## Field: {field_name}
- **Predicted Value**: {predicted_value}
- **Ground Truth Value**: {ground_truth_value}

## Evaluation Criteria
1. **CORRECT**: Prediction exactly matches or is semantically equivalent to ground truth
2. **PARTIALLY_CORRECT**: Prediction captures key information but has minor differences (formatting, abbreviations, partial content)
3. **INCORRECT**: Prediction is wrong, missing when it should exist, or completely different
4. **BOTH_NULL**: Both prediction and ground truth are null/empty (not an error)

## Output Format
Return ONLY valid JSON:
{{
  "rating": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT" | "BOTH_NULL",
  "score": 1.0 | 0.5 | 0.0 | 1.0,
  "rationale": "Brief explanation of your assessment"
}}

Assess the field extraction quality now."""

