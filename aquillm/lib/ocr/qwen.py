"""
Qwen vision model OCR provider.
"""

import base64
import os
import structlog
from typing import Any, Dict

from openai import OpenAI

from .config import get_qwen_config
from .image_utils import get_image_mime_type, resize_image_for_ocr

logger = structlog.stdlib.get_logger(__name__)


def _qwen_ocr_extra_body(model_name: str) -> Dict[str, Any] | None:
    """Return model-specific chat template overrides for OCR requests."""
    configured_model = (
        os.getenv("OCR_VLLM_MODEL")
        or os.getenv("VLLM_MODEL")
        or ""
    ).strip().lower()
    requested_model = (model_name or "").strip().lower()
    if "qwen3.5" in requested_model or "qwen3.5" in configured_model:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def _extract_chat_text_from_completion(response: Any) -> str:
    """Extract text content from OpenAI chat completion response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        return "\n".join(texts).strip()
    return str(content).strip()


def _qwen_ocr_completion(file_content: bytes, prompt: str, max_tokens: int = 2048) -> str:
    """Execute Qwen OCR completion request."""
    base_url, api_key, model, timeout_seconds = get_qwen_config()
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
    extra_body = _qwen_ocr_extra_body(model)
    
    resized_content = resize_image_for_ocr(file_content)
    image_mime = get_image_mime_type(resized_content)
    image_data = base64.b64encode(resized_content).decode("utf-8")
    image_url = f"data:{image_mime};base64,{image_data}"

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are an OCR assistant. Transcribe faithfully and avoid hallucinations.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    return _extract_chat_text_from_completion(response)


def extract_text_with_qwen(file_content: bytes, convert_to_latex: bool = False) -> Dict[str, Any]:
    """Extract text from image using Qwen vision model."""
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
    try:
        extracted_text = _qwen_ocr_completion(file_content, text_prompt, max_tokens=2048)
    except Exception as exc:
        raise ValueError(f"Qwen OCR request failed: {exc}") from exc

    _base_url, _api_key, model_name, _timeout_seconds = get_qwen_config()
    result: Dict[str, Any] = {
        "extracted_text": extracted_text or "NO READABLE TEXT",
        "provider": "qwen",
        "model": model_name,
    }

    if convert_to_latex:
        latex_prompt = """
        Extract equations and convert only math notation to LaTeX.
        Keep normal prose as plain text and preserve line breaks where possible.
        Return only transcription text with inline LaTeX where needed.
        """
        try:
            latex_text = _qwen_ocr_completion(file_content, latex_prompt, max_tokens=2048)
        except Exception as exc:
            raise ValueError(f"Qwen OCR LaTeX request failed: {exc}") from exc
        if latex_text and latex_text != "NO MATH CONTENT":
            result["latex_text"] = latex_text

    return result


__all__ = ['extract_text_with_qwen']
