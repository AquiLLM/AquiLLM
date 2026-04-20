"""Extract and select snippets from recent tool results for compact summaries."""
from __future__ import annotations

import re
from ..types.conversation import Conversation


def select_evidence_snippet(text: str, max_chars: int = 420) -> str:
    from .fallback_heuristics import first_sentence, is_useful_fallback_sentence

    cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text or "")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    sentence_candidates = re.split(r"(?<=[.!?])\s+", cleaned)
    good: list[str] = []
    for sentence in sentence_candidates:
        s = sentence.strip()
        if not is_useful_fallback_sentence(s):
            continue
        good.append(s)
        joined = " ".join(good)
        if len(joined) >= max_chars or len(good) >= 2:
            break
    snippet = " ".join(good).strip()
    if not snippet:
        first = first_sentence(cleaned, max_chars=max_chars)
        if is_useful_fallback_sentence(first):
            snippet = first
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return snippet


def _row_source(row: dict, default: str) -> str:
    title = row.get("title")
    if title is None:
        title = row.get("n")
    source = str(title or "").strip()
    return source or default


def _row_text(row: dict) -> str:
    text = row.get("text")
    if text is None:
        text = row.get("x")
    return str(text or "")


def extract_recent_tool_evidence(
    conversation: Conversation,
    max_snippets: int = 10,
    max_chars_per_snippet: int = 420,
) -> tuple[str, list[tuple[str, str]]]:
    from ..types.messages import ToolMessage, UserMessage

    latest_user_query = ""
    for msg in reversed(conversation.messages):
        if isinstance(msg, UserMessage):
            latest_user_query = (msg.content or "").strip()
            if latest_user_query:
                break

    title_re = re.compile(r"--\s*(.*?)\s*chunk\s*#:", flags=re.IGNORECASE)
    snippets: list[tuple[str, str]] = []
    seen_keys: set[str] = set()

    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]

    for tool_msg in tool_messages[:4]:
        result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
        payload = result_dict.get("result")
        entries: list[tuple[str, str]] = []
        if isinstance(payload, list):
            for row in payload[:12]:
                if not isinstance(row, dict):
                    continue
                entries.append((_row_source(row, tool_msg.tool_name), _row_text(row)))
        elif isinstance(payload, dict):
            entries = [(str(k), str(v)) for k, v in list(payload.items())[:12]]
        elif isinstance(payload, str):
            entries = [(tool_msg.tool_name, payload)]
        for key_text, raw_text in entries:
            source = key_text
            title_match = title_re.search(key_text)
            if title_match and title_match.group(1).strip():
                source = title_match.group(1).strip()
            snippet = select_evidence_snippet(raw_text, max_chars=max_chars_per_snippet)
            if not snippet:
                continue
            dedupe_key = snippet[:180]
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            snippets.append((source, snippet))
            if len(snippets) >= max_snippets:
                break
        if len(snippets) >= max_snippets:
            break

    return latest_user_query, snippets


__all__ = ["extract_recent_tool_evidence", "select_evidence_snippet"]
