"""Embedding generation for document text/image chunks."""
from __future__ import annotations

from os import getenv
from typing import TYPE_CHECKING, Any, Callable, Optional

from tenacity import retry, wait_exponential

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


@retry(wait=wait_exponential())
def get_chunk_embedding(chunk: TextChunk, callback: Optional[Callable[[], None]] = None):
    from aquillm.utils import get_embedding, get_multimodal_embedding

    if chunk.modality == chunk.Modality.IMAGE:
        img_url = image_data_url(chunk)
        caption = multimodal_caption(chunk)
        if img_url:
            chunk.embedding = get_multimodal_embedding(
                prompt=caption,
                image_data_url=img_url,
                input_type="search_document",
            )
        else:
            chunk.embedding = get_embedding(caption, input_type="search_document")
    else:
        chunk.embedding = get_embedding(chunk.content, input_type="search_document")
    if callback:
        callback()


__all__ = [
    "env_int",
    "get_chunk_embedding",
    "image_data_url",
    "image_embedding_payloads",
    "multimodal_caption",
]
