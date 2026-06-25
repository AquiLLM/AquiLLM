"""Evidence-first synthesis for the direct RAG pipeline.

Given a conversation whose last turn is a (synthetic) tool result plus the matching
:class:`~apps.chat.services.rag_evidence.EvidencePacket`, produce a user-facing
assistant answer. This reuses the standard post-tool synthesis machinery
(:meth:`LLMInterface.complete`) so citation enforcement, figure embedding, and
retrieval-notice handling stay consistent with the tool loop, then layers
direct-RAG guarantees on top:

- ``no_results`` returns a transparent retrieval notice without an LLM call.
- Empty/unusable synthesis falls back to an extractive cited summary built from
  the packet (always on for direct RAG, independent of
  ``LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI``).
- Figure requests ensure markdown images when the packet carries image URLs.
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from lib.llm.providers import image_context as imgctx
from lib.llm.providers import visibility
from lib.llm.providers.retrieval_status import document_retrieval_notice
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from apps.chat.services.rag_config import max_figures_per_turn, synthesis_max_tokens
from apps.chat.services.rag_evidence import EvidencePacket

logger = structlog.stdlib.get_logger(__name__)

_MARKDOWN_IMAGE_URL_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_TRAILING_URL_RE = re.compile(r"\(([^)]+)\)$")
_MAX_EXTRACTIVE_POINTS = 5


def _truncate_sentence(text: str, max_chars: int = 240) -> str:
    compact = " ".join((text or "").split()).strip()
    if not compact:
        return ""
    match = re.search(r"(.+?[.!?])(\s|$)", compact)
    sentence = match.group(1) if match else compact
    if len(sentence) > max_chars:
        return sentence[:max_chars].rstrip() + "..."
    return sentence


def _latest_user_message(convo: Conversation) -> UserMessage | None:
    for msg in reversed(convo.messages):
        if isinstance(msg, UserMessage):
            return msg
    return None


def _last_assistant_tool_result(convo: Conversation) -> ToolMessage | None:
    for msg in reversed(convo.messages):
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant":
            return msg
    return None


def _wants_figures(convo: Conversation) -> bool:
    user_message = _latest_user_message(convo)
    if user_message is None:
        return False
    return imgctx.looks_like_image_display_request(user_message.content or "")


def _synthesis_unusable(content: str | None) -> bool:
    visible = visibility.strip_tool_markup(visibility.strip_thinking_blocks(content)).strip()
    if not visible:
        return True
    if visibility.is_interim_assistant_text(visible):
        return True
    if visible == visibility.clean_response_failure_text(after_tool_result=True):
        return True
    return False


def _extractive_summary(packet: EvidencePacket) -> str:
    points: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk in packet.chunks:
        snippet = _truncate_sentence(chunk.get("text") or chunk.get("x") or "")
        if not snippet:
            continue
        citation = chunk.get("citation") or chunk.get("ref")
        key = (snippet, citation)
        if key in seen:
            continue
        seen.add(key)
        points.append(f"- {snippet} {citation}" if citation else f"- {snippet}")
        if len(points) >= _MAX_EXTRACTIVE_POINTS:
            break
    if not points:
        return ""
    header = f'Here is what I found in the {packet.search_scope} for "{packet.query}":'
    return header + "\n" + "\n".join(points)


def _figure_markdown(packet: EvidencePacket) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    limit = max_figures_per_turn()
    for chunk in packet.chunks:
        url = chunk.get("image_url") or chunk.get("u")
        if not isinstance(url, str) or not url or url in seen:
            continue
        if not url.startswith("/aquillm/"):
            continue
        seen.add(url)
        alt = str(chunk.get("title") or chunk.get("text") or "Figure").strip()
        alt = alt.replace("\n", " ")[:80] or "Figure"
        lines.append(f"![{alt}]({url})")
        if len(lines) >= limit:
            return lines
    for url in packet.image_urls:
        if url in seen:
            continue
        seen.add(url)
        lines.append(f"![Figure]({url})")
        if len(lines) >= limit:
            break
    return lines


def _ensure_figures(content: str, packet: EvidencePacket) -> str:
    base = (content or "").rstrip()
    existing = set(_MARKDOWN_IMAGE_URL_RE.findall(base))
    additions: list[str] = []
    for line in _figure_markdown(packet):
        match = _TRAILING_URL_RE.search(line)
        if match and match.group(1) in existing:
            continue
        additions.append(line)
    if not additions:
        return base
    if not base:
        return "\n".join(additions)
    return base + "\n\n" + "\n".join(additions)


def _no_results_notice(convo: Conversation, packet: EvidencePacket) -> str:
    notice = (packet.diagnostic_message or "").strip()
    last_tool = _last_assistant_tool_result(convo)
    if last_tool is not None:
        notice = notice or document_retrieval_notice(last_tool).strip()
    return notice or (
        f'I searched the {packet.search_scope} for "{packet.query}", '
        "but found no relevant passages."
    )


def _replace_last_content(
    convo: Conversation, last: AssistantMessage, new_content: str
) -> Conversation:
    updated = last.model_copy(update={"content": new_content})
    return Conversation(
        system=convo.system, messages=list(convo.messages[:-1]) + [updated]
    )


async def synthesize_from_evidence(
    llm_if: Any,
    convo: Conversation,
    packet: EvidencePacket,
    *,
    stream_func: Any = None,
    max_tokens: int | None = None,
) -> Conversation:
    """Produce the final assistant turn from packaged evidence."""
    if packet.retrieval_status == "no_results" or not packet.chunks:
        return convo + [
            AssistantMessage(
                content=_no_results_notice(convo, packet), stop_reason="end_turn"
            )
        ]

    budget = max_tokens if max_tokens is not None else synthesis_max_tokens()
    result_convo, _ = await llm_if.complete(convo, budget, stream_func=stream_func)

    last = result_convo[-1]
    if not isinstance(last, AssistantMessage) or last.tool_call_id:
        return result_convo

    content = last.content or ""
    new_content = content
    if _synthesis_unusable(content):
        extractive = _extractive_summary(packet)
        if extractive:
            new_content = extractive

    if _wants_figures(convo) and packet.image_urls:
        new_content = _ensure_figures(new_content, packet)

    if new_content != content:
        result_convo = _replace_last_content(result_convo, last, new_content)
    return result_convo


__all__ = ["synthesize_from_evidence"]
