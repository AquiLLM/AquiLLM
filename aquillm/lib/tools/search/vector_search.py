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
    result_items: dict[str, Any] = {}
    has_image_results = False
    for i, chunk in enumerate(results):
        title = titles_by_doc_id.get(chunk.doc_id, "Untitled Document")
        key = f"[Result {i + 1}] -- {title} chunk #: {chunk.chunk_number} chunk_id:{chunk.id}"

        if chunk.modality == image_modality:
            display_url = f"{image_url_prefix}{chunk.doc_id}/"
            has_image_results = True

            result_items[key] = {
                "type": "image",
                "text": truncate(chunk.content),
                "image_url": display_url,
                "doc_id": str(chunk.doc_id),
            }
        else:
            doc_for_chunk = docs_by_doc_id.get(chunk.doc_id)
            if doc_for_chunk is not None and getattr(doc_for_chunk, "image_file", None):
                has_image_results = True
                result_items[key] = {
                    "type": "text_with_image",
                    "text": truncate(chunk.content),
                    "image_url": f"{image_url_prefix}{chunk.doc_id}/",
                    "doc_id": str(chunk.doc_id),
                }
            else:
                result_items[key] = truncate(chunk.content)

    ret: dict[str, Any] = {"result": result_items}
    if has_image_results:
        ret["_image_instruction"] = IMAGE_MARKDOWN_INSTRUCTION

    return ret


__all__ = ["IMAGE_MARKDOWN_INSTRUCTION", "pack_chunk_search_results"]
