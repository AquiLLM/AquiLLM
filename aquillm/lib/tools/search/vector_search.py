"""Format vector / hybrid text-chunk search results for LLM tools (no Django imports)."""
from __future__ import annotations

from typing import Any, Callable, Sequence

IMAGE_MARKDOWN_INSTRUCTION = (
    "One or more results include an image_url. "
    "When discussing those image results, include them in markdown with "
    "![description](image_url) using the image_url field from the result."
)


def pack_chunk_search_results(
    results: Sequence[Any],
    *,
    titles_by_doc_id: dict[Any, str],
    docs_by_doc_id: dict[Any, Any],
    truncate: Callable[[str], str],
    image_modality: Any,
    image_url_prefix: str = "/aquillm/document_image/",
) -> dict[str, Any]:
    """Build the tool `result` dict for multi-chunk search (collections or single doc)."""
    items: list[dict[str, Any]] = []
    has_image_results = False
    for i, chunk in enumerate(results):
        rank = i + 1
        title = titles_by_doc_id.get(chunk.doc_id, "Untitled Document")
        base: dict[str, Any] = {
            "rank": rank,
            "chunk_id": chunk.id,
            "doc_id": str(chunk.doc_id),
            "chunk": chunk.chunk_number,
            "title": title,
        }

        if chunk.modality == image_modality:
            display_url = f"{image_url_prefix}{chunk.doc_id}/"
            has_image_results = True

            items.append(
                {
                    **base,
                    "type": "image",
                    "text": truncate(chunk.content),
                    "image_url": display_url,
                }
            )
        else:
            doc_for_chunk = docs_by_doc_id.get(chunk.doc_id)
            if doc_for_chunk is not None and getattr(doc_for_chunk, "image_file", None):
                has_image_results = True
                items.append(
                    {
                        **base,
                        "type": "text_with_image",
                        "text": truncate(chunk.content),
                        "image_url": f"{image_url_prefix}{chunk.doc_id}/",
                    }
                )
            else:
                items.append({**base, "text": truncate(chunk.content)})

    ret: dict[str, Any] = {"result": items}
    if has_image_results:
        ret["_image_instruction"] = IMAGE_MARKDOWN_INSTRUCTION

    return ret


__all__ = ["IMAGE_MARKDOWN_INSTRUCTION", "pack_chunk_search_results"]
