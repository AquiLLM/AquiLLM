"""User-visible retrieval status helpers for document tool results."""
from __future__ import annotations

from ..types.messages import ToolMessage

DOCUMENT_RETRIEVAL_TOOL_NAMES = {
    "vector_search",
    "search_single_document",
    "more_context",
    "whole_document",
    "document_ids",
}


def document_retrieval_notice(tool_message: ToolMessage) -> str:
    tool_name = str(tool_message.tool_name or "").strip()
    if tool_name not in DOCUMENT_RETRIEVAL_TOOL_NAMES:
        return ""
    result_dict = tool_message.result_dict if isinstance(tool_message.result_dict, dict) else {}
    exception = str(result_dict.get("exception") or "").strip()
    if exception:
        return f"The document search tool failed: {exception}"
    if str(result_dict.get("retrieval_status") or "").strip() == "no_results":
        return str(result_dict.get("retrieval_message") or "").strip()
    return ""


def append_retrieval_notice_if_missing(text: str, notice: str) -> str:
    base = (text or "").rstrip()
    cleaned_notice = (notice or "").strip()
    if not cleaned_notice:
        return base
    if cleaned_notice.lower() in base.lower():
        return base
    if not base:
        return cleaned_notice
    return f"{base}\n\n{cleaned_notice}"


__all__ = ["append_retrieval_notice_if_missing", "document_retrieval_notice"]
