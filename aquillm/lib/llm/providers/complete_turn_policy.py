"""Token budgets and feature policy for completion turns."""
from os import getenv

from ..types.conversation import Conversation
from ..types.messages import ToolMessage

_MECHANICAL_TOOL_MAX_TOKENS = 256


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, value)


def _env_optional_cap(name: str, default: int, minimum: int) -> int:
    """Return a token cap where 0 disables the cap and uses the caller budget."""
    try:
        value = int(getenv(name, str(default)))
    except Exception:
        value = default
    if value <= 0:
        return 0
    return max(minimum, value)


def _conversation_used_whole_document(conversation: Conversation) -> bool:
    for msg in reversed(conversation.messages):
        if isinstance(msg, ToolMessage) and msg.for_whom == "assistant":
            if msg.tool_name in {"whole_document", "search_single_document"}:
                return True
    return False


def _post_tool_output_ceiling() -> int:
    return _env_int("LLM_POST_TOOL_OUTPUT_MAX_TOKENS", 12288, minimum=256)


def _post_tool_global_max(global_max: int) -> int:
    return max(global_max, _post_tool_output_ceiling())


def _resolve_post_tool_max_tokens(
    conversation: Conversation,
    *,
    default_cap: int,
    global_max: int,
) -> int:
    cap = max(default_cap, _post_tool_output_ceiling())
    if _conversation_used_whole_document(conversation):
        cap = max(
            cap,
            _env_int("LLM_POST_TOOL_WHOLE_DOC_MAX_TOKENS", 12288, minimum=256),
        )
    return min(cap, max(global_max, _post_tool_output_ceiling()))


def _resolve_continuation_max_tokens(
    conversation: Conversation,
    *,
    default_cap: int,
    post_tool_budget: int,
    global_max: int,
) -> int:
    cap = default_cap
    if _conversation_used_whole_document(conversation):
        cap = max(
            cap,
            _env_int("LLM_CONTINUATION_WHOLE_DOC_MAX_TOKENS", 6144, minimum=128),
        )
    return min(global_max, post_tool_budget, cap)


def _compact_summary_fallback_enabled() -> bool:
    return getenv("LLM_ALLOW_COMPACT_SUMMARY_FALLBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _extractive_evidence_ui_enabled() -> bool:
    """When false, never replace a failed synthesis with raw chunk/doc bullet dumps."""
    return getenv("LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _post_tool_evidence_retry_enabled() -> bool:
    return getenv("LLM_POST_TOOL_ALLOW_EVIDENCE_RETRY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _post_tool_synthesis_retry_count() -> int:
    return min(4, _env_int("LLM_POST_TOOL_SYNTHESIS_RETRIES", 2, minimum=0))


def _auto_tool_followup_direct_retry_enabled() -> bool:
    return getenv("LLM_AUTO_TOOL_FOLLOWUP_DIRECT_RETRY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _direct_answer_retry_max_tokens() -> int:
    return _env_optional_cap("LLM_DIRECT_ANSWER_RETRY_MAX_TOKENS", 2048, minimum=256)


def _general_answer_max_tokens() -> int:
    return _env_optional_cap("LLM_GENERAL_ANSWER_MAX_TOKENS", 4096, minimum=256)


def _tool_call_retry_max_tokens() -> int:
    return _env_optional_cap("LLM_TOOL_CALL_RETRY_MAX_TOKENS", 2048, minimum=256)


def _resolve_tool_step_max_tokens(max_tokens: int, tool_choice_type: str) -> int:
    requested = _env_optional_cap(
        "LLM_TOOL_STEP_MAX_TOKENS",
        _MECHANICAL_TOOL_MAX_TOKENS,
        minimum=128,
    )
    if requested <= 0:
        requested = _MECHANICAL_TOOL_MAX_TOKENS
    if tool_choice_type == "any":
        retry_cap = _tool_call_retry_max_tokens()
        if retry_cap > 0:
            requested = max(requested, retry_cap)
    return min(max_tokens, requested, _MECHANICAL_TOOL_MAX_TOKENS)


# Shared with the direct-RAG handoff; provider code must not import app services.
DIRECT_SYNTHESIS_GROUNDING = (
    "Selected evidence grounding rules:\n"
    "- The selected evidence in the current tool result is the only factual source "
    "for this answer. Earlier conversation provides request context, not additional "
    "document evidence.\n"
    "- Answer the user's question directly and keep the answer concise, with detail "
    "proportionate to the selected evidence. Do not pad the answer with general "
    "background, recommendations, implications, or explanations beyond the question "
    "and evidence.\n"
    "- Cite every factual claim at the claim. A shared claim or comparison must cite "
    "every supporting paper together at the claim, not only in a sources list. For "
    "each computed comparison, show the calculation or its inputs and cite all input "
    "sources. Identify computed values as calculations, not source-reported results; "
    "do not combine incompatible units or conditions.\n"
    "- Preserve disagreements and each paper's conditions, units, definitions, and "
    "qualifications. Do not invent consensus or a reconciliation.\n"
    "- Distinguish inference from source-reported fact. Contrasting study results "
    "do not establish causation. Do not attribute differences to conditions, "
    "mechanisms, or environmental dependence unless the selected passages explicitly "
    "support that explanation.\n"
    "- If a required paper or fact is absent, state that the selected evidence is "
    "insufficient to answer that part. Never fill the gap from assumptions or prior "
    "conversation claims.\n"
    "- Scope absence and negative evidence to the selected excerpts and the cited "
    "study. One paper not measuring an outcome does not establish that no other "
    "study exists or that no other researcher measured it."
)
