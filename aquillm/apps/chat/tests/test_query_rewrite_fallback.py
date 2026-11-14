"""Rewrite failures retain original authority and versioned query identity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.chat.services.retrieval_query_proposals import build_query_proposals
from apps.chat.tests.query_proposal_test_support import (
    admitted_query_runtimes as admitted_query_runtimes,
)
from apps.chat.tests.query_proposal_test_support import (
    prepare_in_original_scope,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,stop,reason",
    [
        (None, "length", "output_limit"),
        (None, "stop", "empty_response"),
        ("private invalid response", "stop", "invalid_json"),
        ('{"rewrite": 12}', "stop", "invalid_schema"),
    ],
)
async def test_rewrite_failure_preserves_original_binding_and_safe_observation(
    monkeypatch,
    admitted_query_runtimes,
    text,
    stop,
    reason,
):
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("RAG_VOCABULARY_QUERY_ENABLED", "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    runtime = admitted_query_runtimes[0]
    question = "What is A_12?"
    scope, authority, budget = (
        runtime._frozen_retrieval_scope,
        runtime.authorization,
        runtime.budget,
    )
    calls = []

    async def get_message(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=text, stop_reason=stop)

    rows = await prepare_in_original_scope(
        question,
        llm=SimpleNamespace(base_args={}, get_message=get_message),
        runtime=runtime,
    )
    assert [row.text for row in rows] == [question]
    assert runtime._query_original_binding.query == question
    assert runtime._query_original_binding.policy_key == rows[0].policy_key
    assert (
        runtime._frozen_retrieval_scope is scope and runtime.authorization is authority
    )
    assert runtime.budget is budget and budget.planner_calls == len(calls) == 1
    assert budget.actions_used == 0
    observation = runtime.observation["query_proposals"]
    assert observation.get("rewrite_failure_reason") == reason
    assert "private invalid response" not in str(observation)


@pytest.mark.asyncio
async def test_proposal_constraint_rejection_is_not_a_provider_failure(
    monkeypatch,
    admitted_query_runtimes,
):
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", "1")
    monkeypatch.setenv("RAG_VOCABULARY_QUERY_ENABLED", "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    runtime = admitted_query_runtimes[0]

    async def get_message(**kwargs):
        return SimpleNamespace(
            text='{"rewrite":"Models used X in 2024"}', stop_reason="stop"
        )

    question = "Models did NOT use X in 2024"
    rows = await prepare_in_original_scope(
        question,
        llm=SimpleNamespace(base_args={}, get_message=get_message),
        runtime=runtime,
    )
    assert [row.text for row in rows] == [question]
    assert runtime.observation["query_proposals"].get("rewrite_failure_reason") is None
    assert runtime.budget.planner_calls == 1


def test_rewrite_contract_cannot_reuse_the_legacy_policy_identity():
    from apps.collections.services.vocabulary_contracts import digest

    old_key = digest(
        {
            "policy": "faithful-query-proposals-v2",
            "vocabulary": None,
            "source_constraints": (),
        }
    )
    rows = build_query_proposals("What is A_12?", rewrite="Explain A_12?")
    assert [row.kind for row in rows] == ["original", "rewrite"]
    assert rows[0].policy_key == rows[1].policy_key
    assert rows[0].policy_key != old_key


def test_contract_policy_separates_actual_cache_and_publication_binding(
    admitted_query_runtimes,
):
    from apps.chat.services.retrieval_query_publication import query_binding_digest
    from apps.chat.services.retrieval_query_runtime import proposal_scope
    from apps.collections.services.vocabulary_contracts import digest
    from apps.documents.services.chunk_search_scope import (
        bind_search_scope,
        reset_search_scope,
    )
    from apps.documents.services.source_loading import source_runtime_scope

    runtime = admitted_query_runtimes[0]
    row = build_query_proposals("What is A_12?")[0]
    legacy = replace(
        row,
        policy_key=digest(
            {
                "policy": "faithful-query-proposals-v2",
                "vocabulary": None,
                "source_constraints": (),
            }
        ),
    )
    cache_keys, publication_keys = [], []
    with source_runtime_scope(runtime):
        for proposal in (legacy, row):
            scope = replace(runtime._frozen_retrieval_scope)
            with proposal_scope(proposal, runtime=runtime) as bound:
                token = bind_search_scope(scope, proposal.text, graph_enabled=False)
                try:
                    cache_keys.append(scope.cache_key())
                    packet = SimpleNamespace(
                        _publication_query_admission=runtime._query_admission,
                        _publication_query_binding=bound,
                    )
                    publication_keys.append(query_binding_digest(packet))
                finally:
                    reset_search_scope(token)
    assert None not in cache_keys and len(set(cache_keys)) == 2
    assert None not in publication_keys and len(set(publication_keys)) == 2
