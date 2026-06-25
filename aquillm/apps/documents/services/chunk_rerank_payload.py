"""Multimodal document payload shaping for rerank APIs."""
from __future__ import annotations

from typing import Any

from apps.documents.services.chunk_embeddings import image_data_url, multimodal_caption


def rerank_document_payload(chunk: Any) -> Any:
    if chunk.modality != chunk.Modality.IMAGE:
        return chunk.content
    data_url = image_data_url(chunk)
    if not data_url:
        return chunk.content
    caption = multimodal_caption(chunk)
    return [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]


__all__ = ["rerank_document_payload"]
