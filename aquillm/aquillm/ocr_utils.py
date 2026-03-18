import base64
import os
from typing import Dict, Any
import logging
import uuid

import anthropic
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.InternalServerError)),
    reraise=True
)
def extract_text_from_image(image_input, convert_to_latex=False) -> Dict[str, Any]:
    """
    Extract text from an image using Claude API.

    Args:
        image_input: Can be:
            - A string path to an image file
            - A file-like object with read method
            - Bytes containing the image data
        convert_to_latex: Whether to also convert mathematical notation to LaTeX

    Returns:
        Dictionary with extracted_text and optionally latex_text
    """
    result = {}

    try:
        if isinstance(image_input, str) and os.path.exists(image_input):
            with open(image_input, "rb") as f:
                file_content = f.read()
        elif isinstance(image_input, bytes):
            file_content = image_input
        elif hasattr(image_input, 'read'):
            file_content = image_input.read()
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        encoded_image = base64.b64encode(file_content).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Could not process image file: {str(e)}")

    try:
        client = anthropic.Anthropic()

        text_prompt = """This is a STRICT OCR task. Look at the image and ONLY transcribe what is written.

CRITICAL:
- Focus on ONLY extracting text you can clearly see in the image
- NEVER invent or imagine text that isn't there
- If no text is visible, respond with "NO READABLE TEXT"
- DO NOT make guesses about unclear text
- DO NOT add any code snippets
- DO NOT generate anything beyond what is visibly written"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_image}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
        )

        extracted_text = response.content[0].text.strip()
        result["extracted_text"] = extracted_text

        if convert_to_latex:
            latex_prompt = """Extract and convert the equations from this image, paying special attention to vector notation. In physics/math, vectors are often indicated with small bars over letters.

CRITICAL VECTOR NOTATION REQUIREMENTS:
- In this physics/math notes image, vectors are indicated with small bars over letters
- USE ONLY \\bar{} NOTATION, NOT \\vec{} FOR VECTORS
- SPECIFICALLY: Convert ř to $\\bar{r}$ (not $\\vec{r}$)
- SPECIFICALLY: Convert F̄ to $\\bar{F}$ (not $\\vec{F}$)
- SPECIFICALLY: Convert dř to $d\\bar{r}$ (not $d\\vec{r}$)
- Every vector symbol must have a bar in the LaTeX (not an arrow)

OTHER IMPORTANT INSTRUCTIONS:
- Extract BOTH text and mathematics exactly as shown in the image
- Maintain the same line breaks and paragraph structure as the original
- Only convert mathematical notation to LaTeX, leave regular text as plain text
- For integrals with limits, use \\int_{lower}^{upper} (not \\oint)
- For subscripts like v₂, use v_2 in LaTeX
- Use $ symbols to delimit math expressions
- For arrows between points (like 1→2), use $1 \\to 2$ or $W_{1\\to 2}$
- For Greek letters: Σ should be \\Sigma, etc.

EXAMPLES FROM PHYSICS/MATH NOTATION:
- If you see "ř" in the notes, render it as $\\bar{r}$ (not $\\vec{r}$)
- If you see "dř" in the notes, render it as $d\\bar{r}$ (not $d\\vec{r}$)
- If you see "ΣF̄", render it as $\\Sigma\\bar{F}$ (not $\\Sigma\\vec{F}$)
- If you see "v₂" in the notes, render it as $v_2$ (not $v2$)

Go through each equation character by character and ensure every vector has a bar (\\bar{}) notation."""

            latex_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_image}},
                        {"type": "text", "text": latex_prompt},
                    ],
                }],
            )

            latex_text = latex_response.content[0].text.strip()
            if latex_text and latex_text != "NO MATH CONTENT":
                result["latex_text"] = latex_text

        return result

    except Exception as e:
        raise ValueError(f"OCR processing failed: {str(e)}")
