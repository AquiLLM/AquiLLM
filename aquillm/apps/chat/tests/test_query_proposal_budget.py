"""Proposal preparation shares the existing turn and fails back to original."""

from importlib import import_module, util
from types import SimpleNamespace

import pytest

from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
from apps.documents.services.source_loading import SourceRuntime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.chat.tests.query_proposal_test_support import (
    admitted_query_runtimes as admitted_query_runtimes, prepare_in_original_scope,
)


def service():
    name = "apps.chat.services.retrieval_query_runtime"
    assert util.find_spec(name), "bounded query proposal runtime is missing"
    return import_module(name)


@pytest.mark.asyncio
async def test_disabled_proposals_make_no_planner_or_storage_calls(monkeypatch):
    s = service()
    monkeypatch.delenv("RAG_QUERY_REWRITE_ENABLED", raising=False)
    monkeypatch.delenv("RAG_VOCABULARY_QUERY_ENABLED", raising=False)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))
    rows = await s.prepare_query_proposals("Original question", llm=object(), runtime=runtime)
    assert [r.text for r in rows] == ["Original question"]
    assert runtime.budget.planner_calls == runtime.budget.actions_used == 0
    assert getattr(runtime, "_query_vocabulary", None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", ["planner", "actions", "none"])
async def test_model_proposal_uses_original_ledger_and_explicit_provider_bounds(exhausted, monkeypatch, admitted_query_runtimes):
    s = service()
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    runtime = admitted_query_runtimes[0]
    budget = runtime.budget
    if exhausted == "planner":
        assert budget.reserve_planner() and budget.reserve_planner()
    if exhausted == "actions":
        for i in range(3):
            assert budget.reserve_action(str(i))
    sent = []

    async def get_message(**kwargs):
        assert budget.planner_calls == 1
        sent.append(kwargs)
        return SimpleNamespace(text='{"rewrite":"Explain A_12?"}')

    llm = SimpleNamespace(base_args={"model": "synthetic"}, get_message=get_message)
    rows = await prepare_in_original_scope("What is A_12?", llm=llm, runtime=runtime)
    assert rows[0].text == "What is A_12?"
    assert len(rows) == (2 if exhausted == "none" else 1)
    assert len(sent) == (1 if exhausted == "none" else 0)
    if sent:
        assert sent[0]["max_tokens"] == 512
        assert sent[0]["model"] == "synthetic"
    assert budget.planner_calls == (2 if exhausted == "planner" else 0 if exhausted == "actions" else 1)
    assert await prepare_in_original_scope("What is A_12?", llm=llm, runtime=runtime) == rows
    other = await prepare_in_original_scope("Different A_12?", llm=llm, runtime=runtime)
    assert [row.text for row in other] == ["Different A_12?"]
    assert len(sent) == (1 if exhausted == "none" else 0)


@pytest.mark.asyncio
async def test_rejected_and_unavailable_model_outputs_preserve_original(monkeypatch, admitted_query_runtimes):
    s = service()
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    for runtime, output in zip(admitted_query_runtimes, ('{"rewrite":"Models used X in 2024"}', 'not json')):

        async def get_message(**_kwargs):
            return SimpleNamespace(text=output)

        rows = await prepare_in_original_scope("Models did NOT use X in 2024",
            llm=SimpleNamespace(base_args={}, get_message=get_message), runtime=runtime)
        assert [r.text for r in rows] == ["Models did NOT use X in 2024"]
        assert runtime.budget.planner_calls == 1
