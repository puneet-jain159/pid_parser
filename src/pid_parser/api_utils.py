"""
Utilities for handling API responses from vision models.
"""

import re
from typing import Any


def extract_text_from_response(completion: Any) -> str:
    """
    Safely extract text content from Claude API response.
    
    Args:
        completion: API completion object
    
    Returns:
        Extracted text string
    """
    content = completion.choices[0].message.content
    
    # Handle list of content blocks (FMAPI format)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts)
    
    # Handle direct string
    if isinstance(content, str):
        return content
    
    # Fallback
    return str(content) if content else ""


def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from response text, handling markdown code blocks.
    
    Args:
        text: Response text that may contain JSON
    
    Returns:
        Extracted JSON string
    """
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if json_match:
        return json_match.group(1).strip()
    
    # Try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0).strip()
    
    return text.strip()

