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


def replace_payload_text(payload: Any, text: str) -> Any:
    """Replace only caption text, retaining image parts in their original order."""
    if not isinstance(payload, list):
        return text
    return [
        {"type": "text", "text": text}
        if isinstance(part, dict) and part.get("type") == "text"
        else part
        for part in payload
    ]


def append_multimodal_request(payloads, multimodal, documents, model_name, query):
    if any(isinstance(item, list) for item in multimodal):
        payloads.append(
            {
                "model": model_name,
                "query": query,
                "top_n": len(documents),
                "documents": [
                    replace_payload_text(item, document)
                    for item, document in zip(multimodal, documents)
                ],
            }
        )


__all__ = ["rerank_document_payload"]
