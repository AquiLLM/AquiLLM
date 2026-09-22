"""Utilities for enforcing verifiable chunk citations in RAG answers."""
from __future__ import annotations

import re
from os import getenv

from ..types.conversation import Conversation
from ..types.messages import ToolMessage
from .rag_citation_extracts import (
    _chunk_citation_from_row as _chunk_citation_from_row,
)
from .rag_citation_extracts import (
    _doc_ref_from_tool_message as _doc_ref_from_tool_message,
)
from .rag_citation_extracts import (
    _first_citation_token as _first_citation_token,
)
from .rag_citation_extracts import (
    _row_text as _row_text,
)
from .rag_citation_extracts import (
    _snippet_from_whole_document_payload as _snippet_from_whole_document_payload,
)
from .rag_citation_extracts import (
    _truncate_sentence as _truncate_sentence,
)
from .rag_citation_extracts import (
    synthesize_cited_extract_from_results as synthesize_cited_extract_from_results,
)
from .rag_citation_extracts import (
    synthesize_doc_level_extract_from_results as synthesize_doc_level_extract_from_results,
)

_CITATION_RE = re.compile(r"\[doc:[^\]\s]+\s+chunk:\d+\]")
_BULLET_OR_ENUM_RE = re.compile(r"^(\s*[-*]\s+|\s*\d+\.\s+)")


def citation_enforcement_enabled() -> bool:
    value = (getenv("RAG_ENFORCE_CHUNK_CITATIONS", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def citation_sources_append_enabled() -> bool:
    """When false, do not append a trailing ``Sources:`` block (chunk citation list) to replies."""
    value = (getenv("RAG_APPEND_CITATION_SOURCES", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
        result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
        payload = result_dict.get("result")
        rows = list(payload) if isinstance(payload, list) else []
        for row in rows[:max_rows_per_message]:
            if not isinstance(row, dict):
                continue
            citation = _chunk_citation_from_row(row)
            if citation:
                citations.add(citation)
        citation_chunks = result_dict.get("citation_chunks")
        if isinstance(citation_chunks, list):
            for row in citation_chunks:
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


def response_has_required_citations(
    answer_text: str | None,
    allowed_citations: set[str],
    *,
    require_cited_numeric_claims: bool = False,
) -> bool:
    if not allowed_citations:
        return True
    citations = extract_citations(answer_text)
    if not citations:
        return False
    if find_invalid_citations(answer_text, allowed_citations):
        return False
    if find_uncited_factual_lines(answer_text):
        return False
    return not (
        require_cited_numeric_claims and find_uncited_numeric_claims(answer_text)
    )


def find_uncited_numeric_claims(answer_text: str | None) -> list[str]:
    """Find numeric prose without a citation in its sentence or adjoining suffix.

    This checks citation presence, not entailment. Paragraph and list boundaries
    prevent a sources inventory from supplying citations to earlier claims.
    Decimal points and digits inside citation tokens are not sentence boundaries
    or numeric evidence, respectively.
    """
    paragraphs: list[list[str]] = [[]]
    fence: str | None = None
    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith(("```", "~~~")):
            if fence is None:
                fence = line[:3]
            elif line.startswith(fence):
                fence = None
            paragraphs.append([])
            continue
        if fence is not None:
            continue
        source_heading = re.fullmatch(r"(?:\*\*)?Sources:?(?:\*\*)?:?", line, re.I)
        if not line or line.startswith(("#", "![")) or source_heading:
            paragraphs.append([])
            continue
        if _BULLET_OR_ENUM_RE.match(line):
            paragraphs.append([])
            line = _BULLET_OR_ENUM_RE.sub("", line, count=1)
        paragraphs[-1].append(line)

    uncited: list[str] = []
    for lines in paragraphs:
        # A nonnumeric marker also protects punctuation inside document IDs.
        marked = _CITATION_RE.sub(" \x00", " ".join(lines))
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"')\u201d\u2019])\s+", marked)
        claims: list[str] = []
        for sentence in sentences:
            # Accept: "Capacity is 18 litres. [doc:a chunk:1]". The citation
            # belongs to the preceding claim, not an unrelated following one.
            leading_refs = re.match(r"^(?:\s*\x00[\s.,;:]*)+", sentence)
            if leading_refs and claims:
                claims[-1] += " \x00"
                sentence = sentence[leading_refs.end():]
            if sentence.strip():
                claims.append(sentence.strip())
        uncited.extend(
            claim for claim in claims
            if "\x00" not in claim
            and re.search(r"\d", claim)
            and re.search(r"[A-Za-z]{2,}", claim)
        )
    return uncited


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
    inherited_citation: str | None = None
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        if line.startswith("#") or line.startswith("!["):
            continue
        # A cited, non-bullet heading/claim can anchor following indented sub-bullets.
        if not _BULLET_OR_ENUM_RE.match(line):
            cited_here = _first_citation_token(line)
            inherited_citation = cited_here if cited_here else None
            continue
        if line.endswith(":"):
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        words = re.findall(r"[A-Za-z0-9]+", line)
        if len(words) < 3:
            continue
        if inherited_citation and (raw_line[:1].isspace() or raw_line.startswith("\t")):
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


__all__ = [
    "build_citation_retry_prompt",
    "build_citation_system_suffix",
    "citation_enforcement_enabled",
    "citation_sources_append_enabled",
    "collect_allowed_chunk_citations",
    "extract_citations",
    "find_uncited_factual_lines",
    "find_uncited_numeric_claims",
    "find_invalid_citations",
    "response_has_required_citations",
    "synthesize_cited_extract_from_results",
    "synthesize_doc_level_extract_from_results",
]
