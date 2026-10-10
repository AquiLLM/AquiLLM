"""
Multimodal (text + image) embedding support.
"""

import structlog

import requests

from .config import get_local_embed_config
from .provenance import EmbeddingResult, make_result
from .utils import EmbeddingContractError, validate_embedding
from .errors import REQUEST_TIMEOUT_SECONDS, EmbeddingUpstreamUnavailableError, require_transient

logger = structlog.stdlib.get_logger(__name__)


def _check_status(response):
    status = response.status_code
    if status == 200 or status in (400, 404, 405, 415, 422):
        return
    if status in (408, 429) or 500 <= status <= 599:
        raise EmbeddingUpstreamUnavailableError("Multimodal embedding upstream unavailable")
    raise EmbeddingContractError("Multimodal provider rejected the request")


def _response_json(response):
    try:
        return response.json()
    except ValueError:
        raise EmbeddingContractError("Multimodal provider returned malformed JSON") from None


def _single_response_vector(data: object) -> list[float]:
    """Require one unambiguous result for a single text/image input."""
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, (list, tuple)) or len(rows) != 1:
        raise EmbeddingContractError("Multimodal response must contain one vector")
    row = rows[0]
    if not isinstance(row, dict):
        raise EmbeddingContractError("Multimodal response entry is invalid")
    # Some pooling responses omit indices. An explicit index must bind to the
    # only submitted multimodal item; bool/float/string coercion is unsafe.
    if "index" in row and (type(row["index"]) is not int or row["index"] != 0):
        raise EmbeddingContractError("Multimodal response has invalid input index")
    vector = row.get("embedding")
    validate_embedding(vector)
    return list(vector)


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
        instruction = (
            "Represent the given image with the following caption for retrieval."
        )

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


def get_multimodal_embedding_result_via_vllm_pooling(
    prompt: str,
    image_data_url: str,
    input_type: str = "search_document",
) -> EmbeddingResult | None:
    """
    Attempt to get a multimodal embedding via vLLM's native pooling API.

    Uses Qwen3-VL-Embedding format as documented in:
    https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/examples/embedding_vllm.ipynb

    Returns None for unsupported formats; outages and invalid responses propagate.
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
        logger.debug(
            "obs.embed.multimodal_attempt",
            format="multi_modal_data",
            prompt_length=len(formatted_prompt),
        )
        response = requests.post(
            f"{vllm_base}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        _check_status(response)
        if response.status_code == 200:
            data = _response_json(response)
            embedding = _single_response_vector(data)
            logger.info("obs.embed.multimodal_succeeded", format="multi_modal_data")
            return make_result(
                embedding,
                provider="local-openai",
                route="vllm-multi-modal-data",
                role=input_type,
                prepared_input={
                    "input": formatted_prompt,
                    "multi_modal_data": payload["multi_modal_data"],
                },
                original_input={
                    "input": formatted_prompt,
                    "multi_modal_data": payload["multi_modal_data"],
                },
                model=model,
                response_model=data.get("model"),
            )
        else:
            logger.debug(
                "obs.embed.multimodal_non_200",
                format="multi_modal_data",
                status_code=response.status_code,
            )
    except EmbeddingContractError:
        raise
    except Exception as exc:
        require_transient(exc)
        raise EmbeddingUpstreamUnavailableError("Multimodal embedding upstream unavailable") from None

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
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        _check_status(response)
        if response.status_code == 200:
            data = _response_json(response)
            embedding = _single_response_vector(data)
            logger.info("obs.embed.multimodal_succeeded", format="openai_content")
            return make_result(
                embedding,
                provider="local-openai",
                route="vllm-openai-content",
                role=input_type,
                prepared_input=content_payload["input"],
                original_input=content_payload["input"],
                model=model,
                response_model=data.get("model"),
            )
        else:
            logger.debug(
                "obs.embed.multimodal_non_200",
                format="openai_content",
                status_code=response.status_code,
            )
    except EmbeddingContractError:
        raise
    except Exception as exc:
        require_transient(exc)
        raise EmbeddingUpstreamUnavailableError("Multimodal embedding upstream unavailable") from None

    logger.debug("obs.embed.multimodal_unsupported")

    return None


__all__ = [
    "get_multimodal_embedding_via_vllm_pooling",
]


def get_multimodal_embedding_via_vllm_pooling(
    prompt: str, image_data_url: str
) -> list[float] | None:
    result = get_multimodal_embedding_result_via_vllm_pooling(prompt, image_data_url)
    return result.vector if result is not None else None
