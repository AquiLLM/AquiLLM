"""
Multimodal (text + image) embedding support.
"""

import structlog

import requests

from .config import get_local_embed_config

logger = structlog.stdlib.get_logger(__name__)


def _format_qwen_vl_embed_prompt(instruction: str, text: str, has_image: bool) -> str:
    """
    Format a prompt using Qwen3-VL-Embedding's expected chat template.
    
    Based on: https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/embedding_vllm.ipynb
    
    The format is:
    <|im_start|>system
    {instruction}<|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>
    <|im_start|>assistant
    """
    if not instruction:
        instruction = "Represent the given image with the following caption for retrieval."
    
    if has_image:
        user_content = f"<|vision_start|><|image_pad|><|vision_end|>{text}"
    else:
        user_content = text
    
    prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def get_multimodal_embedding_via_vllm_pooling(
    prompt: str,
    image_data_url: str,
) -> list[float] | None:
    """
    Attempt to get a multimodal embedding via vLLM's native pooling API.
    
    Uses Qwen3-VL-Embedding format as documented in:
    https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/embedding_vllm.ipynb
    
    Returns None if multimodal embedding is not supported or fails.
    """
    base_url, api_key, model = get_local_embed_config()
    vllm_base = base_url.rstrip("/")
    if vllm_base.endswith("/v1"):
        vllm_base = vllm_base[:-3]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    formatted_prompt = _format_qwen_vl_embed_prompt(
        instruction="Represent the given image with the following caption for retrieval.",
        text=prompt,
        has_image=True,
    )
    
    # Try /v1/embeddings with multi_modal_data format (vLLM native)
    try:
        payload = {
            "model": model,
            "input": formatted_prompt,
            "multi_modal_data": {
                "image": image_data_url,
            },
        }
        logger.debug("obs.embed.multimodal_attempt", format="multi_modal_data", prompt_length=len(formatted_prompt))
        response = requests.post(
            f"{vllm_base}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0].get("embedding")
                if embedding:
                    logger.info("obs.embed.multimodal_success", format="multi_modal_data")
                    return embedding
        else:
            logger.debug("obs.embed.multimodal_http_error", format="multi_modal_data", status_code=response.status_code, response_text=response.text[:500])
    except Exception as exc:
        logger.debug("obs.embed.multimodal_error", format="multi_modal_data", error_type=type(exc).__name__, error=str(exc))
    
    # Try /v1/embeddings with OpenAI-style content blocks (alternative format)
    try:
        content_payload = {
            "model": model,
            "input": [
                {"type": "text", "text": formatted_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
        logger.debug("obs.embed.multimodal_attempt", format="openai_content")
        response = requests.post(
            f"{vllm_base}/v1/embeddings",
            headers=headers,
            json=content_payload,
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0].get("embedding")
                if embedding:
                    logger.info("obs.embed.multimodal_success", format="openai_content")
                    return embedding
        else:
            logger.debug("obs.embed.multimodal_http_error", format="openai_content", status_code=response.status_code, response_text=response.text[:500])
    except Exception as exc:
        logger.debug("obs.embed.multimodal_error", format="openai_content", error_type=type(exc).__name__, error=str(exc))
    
    logger.debug("obs.embed.multimodal_unsupported")
    
    return None


__all__ = [
    'get_multimodal_embedding_via_vllm_pooling',
]
