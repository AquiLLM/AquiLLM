"""Smoke tests for the offline RAG eval runner (Task 8).

These tests exercise the eval runner logic (case loading, mock dispatch,
pass/fail detection) without any live LLM calls or database access.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apps.chat.evals.run_rag_eval import (
    _build_mock_retrieval,
    _make_consumer,
    _run_case,
    load_cases,
)
from apps.chat.services import rag_pipeline


_CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "rag_cases.yaml"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def test_load_cases_returns_at_least_three():
    cases = load_cases(_CASES_PATH)
    assert len(cases) >= 3


def test_load_cases_have_required_fields():
    cases = load_cases(_CASES_PATH)
    required_fields = {"id", "message", "collection_ids", "expect_outcome"}
    for case in cases:
        missing = required_fields - set(case.keys())
        assert not missing, f"Case {case.get('id')!r} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Explicit search case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_search_case_passes(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    case = {
        "id": "explicit_search_smoke",
        "description": "Explicit search smoke test",
        "message": "Please search the selected documents for calibration notes.",
        "collection_ids": [1],
        "mock_retrieval_status": "results_found",
        "mock_retrieved_count": 1,
        "expect_outcome": "handled",
        "expect_retrieval_called": True,
        "expect_content_contains": ["calibration"],
    }

    result = await _run_case(case, verbose=False)

    assert result["passed"], f"Case failed: {result['failures']}"


# ---------------------------------------------------------------------------
# Collection-backed question case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collection_backed_question_case_passes(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    case = {
        "id": "collection_backed_smoke",
        "description": "Collection-backed question smoke test",
        "message": "What does this paper say about spectral analysis?",
        "collection_ids": [42],
        "mock_retrieval_status": "results_found",
        "mock_retrieved_count": 1,
        "expect_outcome": "handled",
        "expect_retrieval_called": True,
        "expect_content_contains": ["spectral"],
    }

    result = await _run_case(case, verbose=False)

    assert result["passed"], f"Case failed: {result['failures']}"


# ---------------------------------------------------------------------------
# Figure request case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_figure_request_case_passes(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    case = {
        "id": "figure_request_smoke",
        "description": "Figure request smoke test",
        "message": "Show me the figures from the calibration paper.",
        "collection_ids": [1],
        "mock_retrieval_status": "results_found",
        "mock_retrieved_count": 1,
        "expect_outcome": "handled",
        "expect_retrieval_called": True,
        "expect_content_contains": ["calibration"],
    }

    result = await _run_case(case, verbose=False)

    assert result["passed"], f"Case failed: {result['failures']}"


# ---------------------------------------------------------------------------
# No-results case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_results_case_handled_without_llm(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    case = {
        "id": "no_results_smoke",
        "description": "Empty retrieval → notice without LLM",
        "message": "Search the documents for dark matter.",
        "collection_ids": [5],
        "mock_retrieval_status": "no_results",
        "mock_retrieved_count": 0,
        "expect_outcome": "handled",
        "expect_retrieval_called": True,
        "expect_content_contains": ["no relevant passages"],
    }

    result = await _run_case(case, verbose=False)

    assert result["passed"], f"Case failed: {result['failures']}"


# ---------------------------------------------------------------------------
# Non-RAG message should be skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_rag_message_is_skipped(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    case = {
        "id": "non_rag_smoke",
        "description": "General question → skipped",
        "message": "What is the capital of France?",
        "collection_ids": [],
        "mock_retrieval_status": "results_found",
        "mock_retrieved_count": 0,
        "expect_outcome": "skipped",
        "expect_retrieval_called": False,
        "expect_content_contains": [],
    }

    result = await _run_case(case, verbose=False)

    assert result["passed"], f"Case failed: {result['failures']}"


# ---------------------------------------------------------------------------
# All YAML cases pass (integration smoke)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_yaml_cases_pass(monkeypatch):
    """All cases in rag_cases.yaml must pass the eval runner."""
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    cases = load_cases(_CASES_PATH)

    failures = []
    for case in cases:
        result = await _run_case(case, verbose=False)
        if not result["passed"]:
            failures.append(f"[{result['id']}] {result['failures']}")

    assert not failures, "One or more YAML eval cases failed:\n" + "\n".join(failures)
