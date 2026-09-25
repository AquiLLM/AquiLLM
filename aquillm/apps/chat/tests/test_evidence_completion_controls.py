"""Snapshot matching includes actual common completion and publication policy."""

import pytest

from apps.chat.evals.evidence_quality_runtime import comparison_controls
from lib.llm.providers.final_stream import final_answer_streaming_enabled
from lib.llm.synthesis_dispatch import synthesis_limits


def test_demonstrated_lease_and_publication_drift_is_not_snapshot_equivalent(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "0")
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "0")
    before = comparison_controls()
    assert synthesis_limits(4096)["calls"] == 9 and not final_answer_streaming_enabled()
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "4")
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    assert synthesis_limits(4096)["calls"] == 21 and final_answer_streaming_enabled()
    assert comparison_controls() != before


@pytest.mark.parametrize(
    "key,value",
    [
        ("LLM_CONTINUATION_MAX_TOKENS", "7000"),
        ("LLM_POST_TOOL_OUTPUT_MAX_TOKENS", "14000"),
        ("LLM_DIRECT_ANSWER_RETRY_MAX_TOKENS", "3072"),
        ("LLM_CONTINUE_ON_CUTOFF", "0"),
        ("RAG_ENFORCE_CHUNK_CITATIONS", "0"),
        ("RAG_APPEND_CITATION_SOURCES", "0"),
        ("LLM_POST_TOOL_ALLOW_EVIDENCE_RETRY", "0"),
        ("LLM_CITATION_RETRY_PRIOR_MAX_CHARS", "1700"),
    ],
)
def test_common_recovery_output_citation_controls_change_snapshot(
    monkeypatch, key, value
):
    before = comparison_controls()
    monkeypatch.setenv(key, value)
    assert comparison_controls() != before
