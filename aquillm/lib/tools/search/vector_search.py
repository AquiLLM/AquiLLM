"""Format vector / hybrid text-chunk search results for LLM tools (no Django imports)."""
from __future__ import annotations

from os import getenv
from typing import Any, Callable, Sequence

IMAGE_MARKDOWN_INSTRUCTION = (
    "One or more results include an image URL (field `image_url`, or compact payloads: `u` with "
    "`ty` of `image` or `text_with_image`). When discussing those results, include them in markdown "
    "with ![description](url) using that exact URL from the tool result—do not guess document ids."
)


def _compact_items_default() -> bool:
    v = (getenv("TOOL_SEARCH_COMPACT_PAYLOAD", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _doc_has_renderable_image(doc: Any) -> bool:
    image_file = getattr(doc, "image_file", None)
    if not image_file:
        return False
    name = getattr(image_file, "name", "")
    if not name:
        return False
    storage = getattr(image_file, "storage", None)
    exists = getattr(storage, "exists", None) if storage is not None else None
    if callable(exists):
        try:
            return bool(exists(name))
        except Exception:
            return False
    return True


def pack_chunk_search_results(
    results: Sequence[Any],
    *,
    titles_by_doc_id: dict[Any, str],
    docs_by_doc_id: dict[Any, Any],
    truncate: Callable[[str], str],
    image_modality: Any,
    image_url_prefix: str = "/aquillm/document_image/",
    compact_items: bool | None = None,
) -> dict[str, Any]:
    """Build the tool `result` dict for multi-chunk search (collections or single doc)."""
    use_compact = _compact_items_default() if compact_items is None else compact_items
    items: list[dict[str, Any]] = []
    has_image_results = False
    image_renderable_by_doc_id: dict[Any, bool] = {}
    for i, chunk in enumerate(results):
        rank = i + 1
        title = titles_by_doc_id.get(chunk.doc_id, "Untitled Document")

        if use_compact:
            base: dict[str, Any] = {
                "r": rank,
                "i": chunk.id,
                "d": str(chunk.doc_id),
                "c": chunk.chunk_number,
                "n": title,
                "ref": f"[doc:{chunk.doc_id} chunk:{chunk.id}]",
            }
        else:
            base = {
                "rank": rank,
                "chunk_id": chunk.id,
                "doc_id": str(chunk.doc_id),
                "chunk": chunk.chunk_number,
                "title": title,
                "citation": f"[doc:{chunk.doc_id} chunk:{chunk.id}]",
            }

        doc_for_chunk = docs_by_doc_id.get(chunk.doc_id)
        if chunk.doc_id not in image_renderable_by_doc_id:
            image_renderable_by_doc_id[chunk.doc_id] = _doc_has_renderable_image(doc_for_chunk)
        has_renderable_image = image_renderable_by_doc_id[chunk.doc_id]

        if chunk.modality == image_modality:
            if has_renderable_image:
                display_url = f"{image_url_prefix}{chunk.doc_id}/"
                has_image_results = True

                if use_compact:
                    items.append(
                        {
                            **base,
                            "ty": "image",
                            "x": truncate(chunk.content),
                            "u": display_url,
                        }
                    )
                else:
                    items.append(
                        {
                            **base,
                            "type": "image",
                            "text": truncate(chunk.content),
                            "image_url": display_url,
                        }
                    )
            else:
                if use_compact:
                    items.append({**base, "x": truncate(chunk.content)})
                else:
                    items.append({**base, "text": truncate(chunk.content)})
        else:
            if has_renderable_image:
                has_image_results = True
                u = f"{image_url_prefix}{chunk.doc_id}/"
                if use_compact:
                    items.append(
                        {
                            **base,
                            "ty": "text_with_image",
                            "x": truncate(chunk.content),
                            "u": u,
                        }
                    )
                else:
                    items.append(
                        {
                            **base,
                            "type": "text_with_image",
                            "text": truncate(chunk.content),
                            "image_url": u,
                        }
                    )
            else:
                if use_compact:
                    items.append({**base, "x": truncate(chunk.content)})
                else:
                    items.append({**base, "text": truncate(chunk.content)})

    ret: dict[str, Any] = {"result": items}
    if has_image_results:
        ret["_image_instruction"] = IMAGE_MARKDOWN_INSTRUCTION

    return ret


__all__ = ["IMAGE_MARKDOWN_INSTRUCTION", "pack_chunk_search_results"]
