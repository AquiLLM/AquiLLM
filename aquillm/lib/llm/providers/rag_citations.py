"""Utilities for enforcing verifiable chunk citations in RAG answers."""
from __future__ import annotations

import re
from os import getenv
from typing import Any

from ..types.conversation import Conversation
from ..types.messages import ToolMessage

_CITATION_RE = re.compile(r"\[doc:[^\]\s]+\s+chunk:\d+\]")
_BULLET_OR_ENUM_RE = re.compile(r"^(\s*[-*]\s+|\s*\d+\.\s+)")


def citation_enforcement_enabled() -> bool:
    value = (getenv("RAG_ENFORCE_CHUNK_CITATIONS", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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


def collect_allowed_chunk_citations(
    conversation: Conversation,
    *,
    max_tool_messages: int = 4,
    max_rows_per_message: int = 40,
) -> set[str]:
    """Collect valid chunk citation tokens from recent assistant tool results."""
    citations: set[str] = set()
    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]
    for tool_msg in tool_messages[:max_tool_messages]:
        payload = tool_msg.result_dict.get("result") if isinstance(tool_msg.result_dict, dict) else None
        if not isinstance(payload, list):
            continue
        for row in payload[:max_rows_per_message]:
            if not isinstance(row, dict):
                continue
            citation = _chunk_citation_from_row(row)
            if citation:
                citations.add(citation)
    return citations


def extract_citations(answer_text: str | None) -> list[str]:
    if not answer_text:
        return []
    return [match.group(0) for match in _CITATION_RE.finditer(answer_text)]


def find_invalid_citations(answer_text: str | None, allowed_citations: set[str]) -> list[str]:
    seen: set[str] = set()
    invalid: list[str] = []
    for citation in extract_citations(answer_text):
        if citation in seen:
            continue
        seen.add(citation)
        if citation not in allowed_citations:
            invalid.append(citation)
    return invalid


def response_has_required_citations(answer_text: str | None, allowed_citations: set[str]) -> bool:
    if not allowed_citations:
        return True
    citations = extract_citations(answer_text)
    if not citations:
        return False
    if find_invalid_citations(answer_text, allowed_citations):
        return False
    return not find_uncited_factual_lines(answer_text)


def find_uncited_factual_lines(answer_text: str | None) -> list[str]:
    """
    Balanced policy:
    - Require citations for factual list items (bullets/enumerated claims).
    - Allow uncited connective prose between cited claims.
    """
    if not answer_text:
        return []
    uncited: list[str] = []
    in_code_block = False
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        if line.startswith("#") or line.startswith("!["):
            continue
        if line.endswith(":"):
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        if not _BULLET_OR_ENUM_RE.match(line):
            continue
        words = re.findall(r"[A-Za-z0-9]+", line)
        if len(words) < 4:
            continue
        if _first_citation_token(line):
            continue
        uncited.append(line)
    return uncited


def build_citation_system_suffix(allowed_citations: set[str], max_refs: int = 24) -> str:
    """Instruction suffix appended to the system prompt for post-tool synthesis turns."""
    refs = sorted(allowed_citations)[:max_refs]
    refs_text = "\n".join(refs)
    return (
        "When answering from retrieved documents, every factual claim must cite source chunks "
        "using tokens exactly like [doc:<doc_id> chunk:<chunk_id>].\n"
        "Do not invent citations. Use only citations from this allow-list:\n"
        f"{refs_text}"
    )


def build_citation_retry_prompt(
    *,
    prior_answer: str,
    allowed_citations: set[str],
    invalid_citations: list[str] | None = None,
    max_refs: int = 24,
) -> str:
    refs = sorted(allowed_citations)[:max_refs]
    refs_text = "\n".join(refs)
    invalid_text = ""
    if invalid_citations:
        invalid_text = "Invalid citations detected and must be removed: " + ", ".join(invalid_citations) + "\n"
    return (
        "Rewrite your previous answer using only verifiable retrieved evidence.\n"
        "Requirements:\n"
        "- Every factual sentence or bullet must include at least one citation token.\n"
        "- Citation format: [doc:<doc_id> chunk:<chunk_id>]\n"
        "- Use only citations from the allow-list below.\n"
        "- Do not call tools.\n"
        f"{invalid_text}"
        "Allow-list:\n"
        f"{refs_text}\n\n"
        "Previous answer:\n"
        f"{prior_answer}"
    )


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


__all__ = [
    "build_citation_retry_prompt",
    "build_citation_system_suffix",
    "citation_enforcement_enabled",
    "collect_allowed_chunk_citations",
    "extract_citations",
    "find_uncited_factual_lines",
    "find_invalid_citations",
    "response_has_required_citations",
    "synthesize_cited_extract_from_results",
]
