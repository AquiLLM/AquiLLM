"""Source attribution formatting for completed responses."""
import re
from typing import Any

from ..types.conversation import Conversation
from ..types.messages import ToolMessage, UserMessage
from . import image_context as imgctx
from . import rag_citations as citations

_DOC_IMAGE_URL_RE = re.compile(r"/aquillm/document_image/([^/]+)/")


def _build_sources_block(allowed_citations: set[str]) -> str:
    refs = sorted(allowed_citations)
    if not refs:
        return ""
    source_lines = "\n".join(f"- {ref}" for ref in refs)
    return f"Sources:\n{source_lines}"


def _select_source_refs_for_response(
    text: str, allowed_citations: set[str]
) -> set[str]:
    used = {
        c for c in citations.extract_citations(text or "") if c in allowed_citations
    }
    if used:
        return used
    return allowed_citations


def _doc_ref(doc_id: Any) -> str | None:
    if doc_id is None:
        return None
    doc_text = str(doc_id).strip()
    if not doc_text:
        return None
    return f"[doc:{doc_text}]"


def _extract_doc_ref_from_image_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _DOC_IMAGE_URL_RE.search(value)
    if not match:
        return None
    return _doc_ref(match.group(1))


def _collect_doc_refs_from_embedded_images(text: str) -> set[str]:
    refs: set[str] = set()
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text or ""):
        ref = _extract_doc_ref_from_image_url(match.group(1))
        if ref:
            refs.add(ref)
    return refs


def _latest_user_requested_image(conversation: Conversation) -> bool:
    for msg in reversed(conversation.messages):
        if isinstance(msg, UserMessage):
            return imgctx.looks_like_image_display_request(msg.content)
    return False


def _collect_source_refs_from_tool_message(tool_message: ToolMessage) -> set[str]:
    refs: set[str] = set()

    if isinstance(tool_message.arguments, dict):
        direct_doc_ref = _doc_ref(tool_message.arguments.get("doc_id"))
        if direct_doc_ref:
            refs.add(direct_doc_ref)

    result_dict = (
        tool_message.result_dict if isinstance(tool_message.result_dict, dict) else {}
    )
    payload = result_dict.get("result")
    payload_rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        payload_rows = [payload]
    elif isinstance(payload, list):
        payload_rows = [row for row in payload if isinstance(row, dict)]

    for row in payload_rows[:40]:
        row_doc_ref = _doc_ref(row.get("doc_id") or row.get("d"))
        if row_doc_ref:
            refs.add(row_doc_ref)
        image_url_ref = _extract_doc_ref_from_image_url(
            row.get("image_url") or row.get("u")
        )
        if image_url_ref:
            refs.add(image_url_ref)

    if not refs and isinstance(payload, str):
        image_url_ref = _extract_doc_ref_from_image_url(payload)
        if image_url_ref:
            refs.add(image_url_ref)

    return refs


def _append_citation_sources_if_missing(
    text: str,
    allowed_citations: set[str],
) -> str:
    base = (text or "").rstrip()
    if not allowed_citations:
        return base
    source_refs = _select_source_refs_for_response(base, allowed_citations)
    sources_block = _build_sources_block(source_refs)
    if not sources_block:
        return base
    sources_index = base.rfind("Sources:")
    if sources_index >= 0:
        existing_sources = base[sources_index:]
        if any(ref in existing_sources for ref in source_refs):
            return base
        source_lines = "\n".join(f"- {ref}" for ref in sorted(source_refs))
        return f"{base}\n{source_lines}"
    return f"{base}\n\n{sources_block}"
