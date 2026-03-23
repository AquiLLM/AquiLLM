"""Compact evidence summarization for post-tool LLM turns."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..types.messages import UserMessage
from . import fallback_heuristics as fb
from .tool_evidence import extract_recent_tool_evidence

if TYPE_CHECKING:
    from .base import LLMInterface
    from ..types.conversation import Conversation


async def generate_compact_tool_summary(
    llm: LLMInterface,
    conversation: Conversation,
    max_tokens: int,
) -> Optional[str]:
    latest_user_query, evidence = extract_recent_tool_evidence(conversation)
    if not evidence:
        return None

    evidence_lines = "\n".join(
        [f"{i + 1}. [{source}] {snippet}" for i, (source, snippet) in enumerate(evidence[:8])]
    )
    user_request = latest_user_query or "Summarize the key points from these retrieved document excerpts."
    summary_prompt = (
        f"User request: {user_request}\n\n"
        "You are given evidence snippets retrieved from documents.\n"
        "Write a complete, direct answer for the user.\n"
        "Requirements:\n"
        "- Provide 4 to 8 concise key points.\n"
        "- Use normal readable prose and bullets.\n"
        "- Do not mention tools, retrieval, or internal system behavior.\n"
        "- Prefer conceptual conclusions and practical takeaways over raw benchmark tables.\n"
        "- If numbers are included, explain what they mean.\n"
        "- If evidence is incomplete, state what is missing in one sentence.\n\n"
        f"Evidence snippets:\n{evidence_lines}"
    )
    summary_max_tokens = max(512, min(max_tokens + 256, 1400))
    attempt_prompt = summary_prompt
    best_text: Optional[str] = None
    for _ in range(3):
        try:
            summary_response = await llm.get_message(
                system="You summarize technical evidence into a clear final answer.",
                messages=[{"role": "user", "content": attempt_prompt}],
                messages_pydantic=[UserMessage(content=attempt_prompt)],
                max_tokens=summary_max_tokens,
            )
        except Exception:
            continue

        text = (summary_response.text or "").strip()
        if not text or fb.looks_like_deferred_tool_intent(text):
            continue
        best_text = text
        stop_reason_normalized = str(summary_response.stop_reason or "").strip().lower()
        if (
            stop_reason_normalized in {"length", "max_tokens"}
            or fb.looks_cut_off(text)
            or not fb.is_high_quality_summary(text)
        ):
            attempt_prompt = (
                f"{summary_prompt}\n\n"
                "The previous draft was incomplete or low quality. "
                "Rewrite the answer as a polished final response with coherent bullets and clear conclusions.\n\n"
                f"Draft to improve:\n{text}"
            )
            continue
        return text
    if best_text and fb.is_high_quality_summary(best_text):
        return best_text
    if best_text and len(best_text.strip()) >= 140 and not fb.looks_like_deferred_tool_intent(best_text):
        return best_text.strip()
    return None


__all__ = ["generate_compact_tool_summary"]
