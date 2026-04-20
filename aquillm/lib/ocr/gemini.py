"""
Google Gemini OCR provider.
"""

import base64
import json
import structlog
import threading
from typing import Any, Dict

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import google.generativeai as genai

from .config import get_gemini_api_key, log_usage

logger = structlog.stdlib.get_logger(__name__)


class GeminiCostTracker:
    """Thread-safe cost tracker for Gemini API usage."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_calls = 0
        self.lock = threading.Lock()

        self.input_cost_per_1k = 0.0005
        self.output_cost_per_1k = 0.0015

    def add_usage(self, input_tokens: int, output_tokens: int) -> float:
        with self.lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.api_calls += 1

            input_cost = (input_tokens / 1000) * self.input_cost_per_1k
            output_cost = (output_tokens / 1000) * self.output_cost_per_1k
            call_cost = input_cost + output_cost
            self.total_cost += call_cost
            return call_cost

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "total_cost_usd": self.total_cost,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "api_calls": self.api_calls,
            }


cost_tracker = GeminiCostTracker()


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    reraise=True,
)
def extract_text_with_gemini(file_content: bytes, convert_to_latex: bool = False) -> Dict[str, Any]:
    """Extract text from image using Google Gemini."""
    result: Dict[str, Any] = {"provider": "gemini", "model": "gemini-1.5-pro"}
    encoded_image = base64.b64encode(file_content).decode("utf-8")

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={
            "temperature": 0.0,
            "top_p": 0.5,
            "top_k": 10,
            "max_output_tokens": 2048,
        },
    )

    text_prompt = """
    This is a STRICT OCR task. Look at the image and ONLY transcribe what is written.

    CRITICAL:
    - Focus on ONLY extracting text you can clearly see in the image
    - NEVER invent or imagine text that isn't there
    - If no text is visible, respond with "NO READABLE TEXT"
    - DO NOT make guesses about unclear text
    - DO NOT add any code snippets
    - DO NOT generate anything beyond what is visibly written
    """

    content_parts = [
        {"text": text_prompt},
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded_image,
            }
        },
    ]

    response = model.generate_content(content_parts)
    extracted_text = response.text.strip()
    result["extracted_text"] = extracted_text

    if hasattr(response, "usage"):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.candidates_tokens
    else:
        input_tokens = len(json.dumps(content_parts)) // 4
        output_tokens = len(extracted_text) // 4

    log_usage("OCR", input_tokens, output_tokens)
    cost_tracker.add_usage(input_tokens=input_tokens, output_tokens=output_tokens)

    if convert_to_latex:
        latex_prompt = """
        Extract equations and convert only math notation to LaTeX.
        Keep normal prose as plain text and preserve line breaks where possible.
        """

        latex_content_parts = [
            {"text": latex_prompt},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encoded_image,
                }
            },
        ]
        latex_response = model.generate_content(latex_content_parts)
        latex_text = latex_response.text.strip()
        if latex_text and latex_text != "NO MATH CONTENT":
            result["latex_text"] = latex_text

            if hasattr(latex_response, "usage"):
                latex_input_tokens = latex_response.usage.prompt_tokens
                latex_output_tokens = latex_response.usage.candidates_tokens
            else:
                latex_input_tokens = len(json.dumps(latex_content_parts)) // 4
                latex_output_tokens = len(latex_text) // 4

            log_usage("LaTeX Conversion", latex_input_tokens, latex_output_tokens)
            cost_tracker.add_usage(
                input_tokens=latex_input_tokens,
                output_tokens=latex_output_tokens,
            )

    return result


__all__ = ['extract_text_with_gemini', 'GeminiCostTracker', 'cost_tracker']
