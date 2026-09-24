"""Explicit document capacity and remaining request-context allocation."""

from os import getenv

from apps.chat.services.rag_config import evidence_token_budget, synthesis_max_tokens
from lib.llm.evidence_guard import estimate_request_tokens
from lib.llm.providers.openai_tokens import context_reserve_tokens, env_int
from lib.llm.types.messages import ToolMessage
from lib.llm.utils.prompt_budget import (
    prompt_budget_context_limit,
    prompt_budget_slack_tokens,
)

SYNTHESIS_INSTRUCTION = (
    "Final synthesis step: give a concise answer with detail proportionate "
    "to the selected evidence. Complete the supported answer without "
    "padding it. Do not emit status lines, tool markup, or promises to retrieve later."
)


def available_evidence_tokens(
    *, model_context, prompt_tokens, output_reserve, safety_margin, configured_budget
):
    return max(
        0,
        min(
            configured_budget,
            model_context - prompt_tokens - output_reserve - safety_margin,
        ),
    )


def resolve_document_cap(*, mode, legacy_cap, explicit_hard_cap, final_passage_limit):
    if (
        mode not in ("legacy", "budgeted")
        or min(legacy_cap, final_passage_limit) <= 0
        or explicit_hard_cap < 0
    ):
        raise ValueError("invalid document capacity")
    return min(
        final_passage_limit,
        legacy_cap if mode == "legacy" else (explicit_hard_cap or final_passage_limit),
    )


def provider_context_capacity(llm):
    provider = type(llm).__name__.lower()
    key = (
        "GEMINI_CONTEXT_LIMIT"
        if "gemini" in provider
        else "CLAUDE_CONTEXT_LIMIT"
        if "claude" in provider
        else "OPENAI_CONTEXT_LIMIT"
    )
    try:
        context = int(getenv(key, "0")) or prompt_budget_context_limit()
    except ValueError:
        context = 0
    margin = (
        sum(context_reserve_tokens(context))
        + max(
            384,
            prompt_budget_slack_tokens(),
            env_int("OPENAI_COMPAT_PROMPT_SLACK_TOKENS", 256),
            env_int("OPENAI_API_PROMPT_SLACK_TOKENS", 384),
        )
        + 256
    )
    return max(0, context), margin


def synthesis_request_skeleton(conversation, rows):
    from lib.llm.providers.complete_turn_policy import DIRECT_SYNTHESIS_GROUNDING
    from lib.llm.providers.rag_citations import (
        _chunk_citation_from_row,
        build_citation_system_suffix,
    )

    request = conversation.model_copy(deep=True)
    for message in request.messages:
        if isinstance(message, ToolMessage):
            message.content = (
                "Earlier tool evidence omitted for this synthesis request."
            )
            message.result_dict = {}
            message.files = None
    citations = {c for row in rows if (c := _chunk_citation_from_row(row))}
    system = (
        request.system
        + "\n\n"
        + DIRECT_SYNTHESIS_GROUNDING
        + "\n\n"
        + SYNTHESIS_INSTRUCTION
    )
    if citations:
        system += "\n\n" + build_citation_system_suffix(citations)
    tools = (
        [tool.llm_definition for tool in (getattr(request[-1], "tools", None) or [])]
        if request.messages
        else []
    )
    # Selected row bodies/metadata are charged by candidate_token_cost. Reserve
    # the result envelope, tool-call wrapper and request framing here as well.
    return {
        "system": system,
        "messages": [m.render(include={"role", "content"}) for m in request.messages],
        "tools": tools,
        "selected_tool_result": {
            "result": [],
            "retrieval_status": "results_found",
            "retrieved_count": len(rows),
            "retrieved_documents": [],
        },
    }


def synthesis_evidence_budget(conversation, llm, rows, *, output_reserve=None):
    context, margin = provider_context_capacity(llm)
    if conversation is None:
        return 0
    from apps.documents.services.source_loading import current_source_runtime

    runtime = current_source_runtime()
    prompt = estimate_request_tokens(
        synthesis_request_skeleton(conversation, rows),
        turn_budget=runtime.budget if runtime else None,
    )
    return available_evidence_tokens(
        model_context=context,
        prompt_tokens=prompt,
        output_reserve=output_reserve
        if output_reserve is not None
        else synthesis_max_tokens(),
        safety_margin=margin,
        configured_budget=evidence_token_budget(),
    )
