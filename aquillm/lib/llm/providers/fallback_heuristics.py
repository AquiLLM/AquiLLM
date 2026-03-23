"""Heuristics for extractive fallback and deferred-tool detection."""
from __future__ import annotations

import re
from os import getenv
from typing import Optional

from ..types.conversation import Conversation
from ..types.messages import ToolMessage


def looks_like_deferred_tool_intent(text: Optional[str]) -> bool:
    """
    Heuristic: detect when the model says it will search/look something up
    instead of actually issuing a tool call in this turn.
    """
    if not text:
        return False
    normalized = re.sub(r"[\u2018\u2019]", chr(39), text.lower())
    cues = (
        "i'll search",
        "i will search",
        "i'll search for",
        "let me search",
        "i can search",
        "i'm going to search",
        "i'll look through",
        "i will look through",
        "i'll look up",
        "i will look up",
        "let me look up",
        "i'll check",
        "i will check",
        "i'll read the papers",
        "i will read the papers",
    )
    return any(cue in normalized for cue in cues)


def extractive_fallback_enabled() -> bool:
    return getenv("LLM_ALLOW_EXTRACTIVE_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")


def first_sentence(text: str, max_chars: int = 260) -> str:
    cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text or "")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(\s|$)", cleaned)
    candidate = match.group(1) if match else cleaned
    if len(candidate) > max_chars:
        return candidate[:max_chars].rstrip() + "..."
    return candidate


def is_useful_fallback_sentence(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    if not re.match(r'^[A-Z"(]', candidate):
        return False
    words = re.findall(r"[A-Za-z0-9]+", candidate)
    if len(words) < 10:
        return False
    alpha_chars = len(re.findall(r"[A-Za-z]", candidate))
    if alpha_chars < 40:
        return False
    digit_chars = len(re.findall(r"\d", candidate))
    if digit_chars / max(1, len(candidate)) > 0.08:
        return False
    upper_chars = len(re.findall(r"[A-Z]", candidate))
    if alpha_chars and (upper_chars / alpha_chars) > 0.45:
        return False
    bad_tokens = ("mmlu", "bbh", "gsm8k", "triviaqa", "humaneval", "mbpp", "cmath")
    lowered = candidate.lower()
    if sum(1 for token in bad_tokens if token in lowered) >= 2:
        return False
    return True


def is_high_quality_summary(text: str) -> bool:
    candidate = (text or "").strip()
    if len(candidate) < 220:
        return False
    lowered = candidate.lower()
    if lowered.startswith("here are the key points from the retrieved passages"):
        return False
    if "i retrieved supporting passages but could not generate a final answer" in lowered:
        return False
    if "please retry and i will provide a direct summary" in lowered:
        return False
    bad_tokens = ("mmlu", "bbh", "gsm8k", "triviaqa", "humaneval", "mbpp", "cmath")
    if sum(1 for token in bad_tokens if token in lowered) >= 5:
        return False
    bullet_count = candidate.count("\n- ") + candidate.count("\n* ")
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", candidate))
    return bullet_count >= 3 or sentence_count >= 5


def looks_cut_off(text: str) -> bool:
    cleaned = (text or "").rstrip()
    if not cleaned:
        return False
    if cleaned.endswith(("...", "…")):
        return True
    if cleaned[-1] in ".!?)]}\"'":
        return False
    return True


def continue_on_cutoff_enabled() -> bool:
    return getenv("LLM_CONTINUE_ON_CUTOFF", "1").strip().lower() in ("1", "true", "yes", "on")


def synthesize_from_recent_tool_results(conversation: Conversation) -> Optional[str]:
    tool_messages = [
        msg
        for msg in reversed(conversation.messages)
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant"
    ]
    if not tool_messages:
        return None

    bullets: list[str] = []
    seen: set[str] = set()
    source_titles: list[str] = []
    title_re = re.compile(r"--\s*(.*?)\s*chunk\s*#:", flags=re.IGNORECASE)

    for tool_msg in tool_messages[:3]:
        result_dict = tool_msg.result_dict if isinstance(tool_msg.result_dict, dict) else {}
        payload = result_dict.get("result")
        if isinstance(payload, dict):
            for k, v in list(payload.items())[:8]:
                key_text = str(k)
                val_text = str(v)
                title_match = title_re.search(key_text)
                if title_match:
                    title = title_match.group(1).strip()
                    if title and title not in source_titles:
                        source_titles.append(title)
                sentence = first_sentence(val_text)
                if sentence and is_useful_fallback_sentence(sentence) and sentence not in seen:
                    seen.add(sentence)
                    bullets.append(sentence)
                if len(bullets) >= 6:
                    break
        elif isinstance(payload, str):
            sentence = first_sentence(payload)
            if sentence and is_useful_fallback_sentence(sentence) and sentence not in seen:
                seen.add(sentence)
                bullets.append(sentence)
        if len(bullets) >= 6:
            break

    if not bullets:
        return None

    header = "Here are the key points from the retrieved passages:"
    bullet_lines = [f"- {point}" for point in bullets[:5]]
    if source_titles:
        sources = ", ".join(source_titles[:4])
        return f"{header}\n" + "\n".join(bullet_lines) + f"\n\nSources consulted: {sources}"
    return f"{header}\n" + "\n".join(bullet_lines)


__all__ = [
    "continue_on_cutoff_enabled",
    "extractive_fallback_enabled",
    "first_sentence",
    "is_high_quality_summary",
    "is_useful_fallback_sentence",
    "looks_cut_off",
    "looks_like_deferred_tool_intent",
    "synthesize_from_recent_tool_results",
]
