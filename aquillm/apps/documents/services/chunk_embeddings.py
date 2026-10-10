"""Embedding generation for document text/image chunks."""

from __future__ import annotations

from collections.abc import Callable
from os import getenv
from typing import TYPE_CHECKING, Any

from tenacity import retry, retry_if_not_exception_type, wait_exponential

from lib.embeddings.utils import EmbeddingContractError

if TYPE_CHECKING:
    from apps.documents.models.chunks import TextChunk


def env_int(name: str, default: int) -> int:
    try:
        value = int((getenv(name) or str(default)).strip())
    except Exception:
        value = default
    return value if value > 0 else default


def multimodal_caption(chunk: TextChunk) -> str:
    char_limit = env_int("APP_RAG_IMAGE_CAPTION_CHAR_LIMIT", 800)
    text = (chunk.content or "").strip()
    if not text:
        text = "Image chunk"
    return text[:char_limit]


def image_data_url(chunk: TextChunk) -> str | None:
    if chunk.modality != chunk.Modality.IMAGE:
        return None
    try:
        doc = chunk.document
    except Exception:
        return None
    from apps.documents.services.image_payloads import doc_image_data_url

    return doc_image_data_url(doc)


def image_embedding_payloads(chunk: TextChunk) -> list[Any]:
    data_url = image_data_url(chunk)
    if not data_url:
        return []
    caption = multimodal_caption(chunk)
    return [
        [
            {"type": "input_text", "text": caption},
            {"type": "input_image", "image_url": data_url},
        ],
        [
            {"type": "input_text", "text": caption},
            {"type": "input_image", "image_url": {"url": data_url}},
        ],
        [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
        [{"type": "input_image", "image_url": data_url}],
    ]


@retry(
    wait=wait_exponential(),
    retry=retry_if_not_exception_type(EmbeddingContractError),
)
def get_chunk_embedding(chunk: TextChunk, callback: Callable[[], None] | None = None):
    from aquillm.utils import get_embedding_result, get_multimodal_embedding_result

    if chunk.modality == chunk.Modality.IMAGE:
        img_url = image_data_url(chunk)
        caption = multimodal_caption(chunk)
        if img_url:
            result = get_multimodal_embedding_result(
                prompt=caption,
                image_data_url=img_url,
                input_type="search_document",
            )
        else:
            result = get_embedding_result(caption, input_type="search_document")
    else:
        result = get_embedding_result(chunk.content, input_type="search_document")
    chunk.embedding = result.vector
    chunk.embedding_provenance = result.provenance
    if callback:
        callback()


__all__ = [
    "env_int",
    "get_chunk_embedding",
    "image_data_url",
    "image_embedding_payloads",
    "multimodal_caption",
]
