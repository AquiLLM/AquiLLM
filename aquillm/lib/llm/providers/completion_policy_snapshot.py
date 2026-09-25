"""Observe common completion policy through the getters used by orchestration."""

from . import complete_turn_policy as policy
from .fallback_heuristics import continue_on_cutoff_enabled, extractive_fallback_enabled
from .final_stream import final_answer_streaming_enabled
from .rag_citations import citation_enforcement_enabled, citation_sources_append_enabled


def completion_policy_snapshot(*, output_tokens):
    from lib.llm.synthesis_dispatch import synthesis_limits

    getters = (
        "post_tool_output_ceiling",
        "whole_document_output_cap",
        "whole_document_continuation_cap",
        "tool_step_token_cap",
        "compact_summary_fallback_enabled",
        "extractive_evidence_ui_enabled",
        "post_tool_evidence_retry_enabled",
        "post_tool_synthesis_retry_count",
        "auto_tool_followup_direct_retry_enabled",
        "direct_answer_retry_max_tokens",
        "general_answer_max_tokens",
        "tool_call_retry_max_tokens",
    )
    return {
        **{name: getattr(policy, "_" + name)() for name in getters},
        "turn_token_limits": list(policy.turn_token_limits()),
        "mechanical_tool_max_tokens": policy._MECHANICAL_TOOL_MAX_TOKENS,
        "continue_on_cutoff": continue_on_cutoff_enabled(),
        "extractive_fallback": extractive_fallback_enabled(),
        "final_answer_only": final_answer_streaming_enabled(),
        "enforce_citations": citation_enforcement_enabled(),
        "append_sources": citation_sources_append_enabled(),
        "synthesis_lease": synthesis_limits(output_tokens),
    }
