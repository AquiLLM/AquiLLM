"""Build deterministic cited extracts from retrieved tool evidence."""
from __future__ import annotations

import re
from typing import Any

from ..types.conversation import Conversation
from ..types.messages import ToolMessage

_CITATION_RE = re.compile(r"\[doc:[^\]\s]+\s+chunk:\d+\]")


def _first_citation_token(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    match = _CITATION_RE.search(text.strip())
    if not match:
        return None
    return match.group(0)


def _chunk_citation_from_row(row: dict[str, Any]) -> str | None:
    for key in ("citation", "ref"):
        citation = _first_citation_token(row.get(key))
        if citation:
            return citation
    doc_id = row.get("doc_id")
    if doc_id is None:
        doc_id = row.get("d")
    chunk_id = row.get("chunk_id")
    if chunk_id is None:
        chunk_id = row.get("i")
    if doc_id is None or chunk_id is None:
        return None
    doc_text = str(doc_id).strip()
    chunk_text = str(chunk_id).strip()
    if not doc_text or not chunk_text.isdigit():
        return None
    return f"[doc:{doc_text} chunk:{chunk_text}]"


def _row_text(row: dict[str, Any]) -> str:
    text = row.get("text")
    if text is None:
        text = row.get("x")
    return str(text or "").strip()


def _truncate_sentence(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split()).strip()
    if not compact:
        return ""
    match = re.search(r"(.+?[.!?])(\s|$)", compact)
    sentence = match.group(1) if match else compact
    if len(sentence) > max_chars:
        return sentence[:max_chars].rstrip() + "..."
    return sentence


def synthesize_cited_extract_from_results(
    conversation: Conversation,
    *,
    max_points: int = 5,
) -> str | None:
    """Extract short, directly-cited bullets from recent chunk search results."""
    points: list[str] = []
    seen: set[str] = set()
    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]
    for tool_msg in tool_messages[:4]:
        payload = tool_msg.result_dict.get("result") if isinstance(tool_msg.result_dict, dict) else None
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            citation = _chunk_citation_from_row(row)
            if not citation:
                continue
            snippet = _truncate_sentence(_row_text(row))
            if not snippet:
                continue
            dedupe_key = f"{snippet}|{citation}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            points.append(f"- {snippet} {citation}")
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break
    if not points:
        return None
    return "I can only provide claims directly supported by retrieved chunks:\n" + "\n".join(points)


def _doc_ref_from_tool_message(tool_msg: ToolMessage) -> str | None:
    if isinstance(tool_msg.arguments, dict):
        doc_id = tool_msg.arguments.get("doc_id")
        if doc_id is not None:
            doc_text = str(doc_id).strip()
            if doc_text:
                return f"[doc:{doc_text}]"
    payload = tool_msg.result_dict.get("result") if isinstance(tool_msg.result_dict, dict) else None
    if isinstance(payload, dict):
        doc_id = payload.get("doc_id") or payload.get("d")
        if doc_id is not None:
            doc_text = str(doc_id).strip()
            if doc_text:
                return f"[doc:{doc_text}]"
    return None


def _snippet_from_whole_document_payload(payload: Any, *, max_chars: int = 220) -> str:
    from .tool_evidence import select_evidence_snippet

    text = ""
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, dict):
        raw = payload.get("text")
        if raw is None:
            raw = payload.get("full_text")
        if raw is None and payload.get("type") == "image_document":
            raw = payload.get("title")
        text = str(raw or "")
    snippet = select_evidence_snippet(text, max_chars=max_chars)
    if snippet:
        return snippet
    return _truncate_sentence(text, max_chars=max_chars)


def synthesize_doc_level_extract_from_results(
    conversation: Conversation,
    *,
    max_points: int = 5,
) -> str | None:
    """Bullets from whole-document / document_with_figures tool payloads (doc-level cites)."""
    points: list[str] = []
    seen: set[str] = set()
    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]
    for tool_msg in tool_messages[:4]:
        if tool_msg.tool_name not in {"whole_document", "search_single_document"}:
            continue
        result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
        if result_dict.get("exception"):
            continue
        payload = result_dict.get("result")
        doc_ref = _doc_ref_from_tool_message(tool_msg)
        if isinstance(payload, list):
            continue
        snippet = _snippet_from_whole_document_payload(payload)
        if not snippet:
            continue
        dedupe_key = f"{snippet}|{doc_ref or tool_msg.tool_name}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        suffix = f" {doc_ref}" if doc_ref else ""
        points.append(f"- {snippet}{suffix}")
        if len(points) >= max_points:
            break
    if not points:
        return None
    return (
        "Here is a concise summary from the retrieved document"
        + ("s" if len(points) > 1 else "")
        + ":\n"
        + "\n".join(points)
    )
