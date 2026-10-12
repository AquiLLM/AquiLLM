"""Service-composition probes use genuine admission and one original ledger."""
from dataclasses import replace
import asyncio

import pytest
import pytest_asyncio

from apps.chat.services.rag_acquisition import acquire_evidence
from apps.chat.services.retrieval_query_binding import ensure_admission
from apps.chat.tests.query_proposal_test_support import admitted_query_runtimes as admitted_query_runtimes
from apps.chat.services.retrieval_query_proposals import build_query_proposals
from apps.chat.services.retrieval_query_runtime import proposal_scope, proposal_identity
from apps.chat.tests.test_rag_selection_scoring import Policy, _authorization
from apps.collections.services.vocabulary_contracts import VocabularyAliasV1, VocabularySnapshotV1
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest_asyncio.fixture(autouse=True)
async def close_fixture_worker_connection():
    from channels.db import database_sync_to_async
    from django.db import connections
    yield
    await database_sync_to_async(connections.close_all)()


def runtime():
    return SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_three_proposal_actions_leave_no_iterative_planner_call():
    from channels.db import database_sync_to_async
    from apps.collections.tests.test_vocabulary_scope import setup_scope, publish, service, entry
    from apps.chat.services.retrieval_query_runtime import _frozen_vocabulary
    actor, collection, authorization = await database_sync_to_async(setup_scope)()
    _, version = await database_sync_to_async(publish)(actor, collection, [entry("A_12", "item")])
    await database_sync_to_async(service().attach_vocabulary)(actor=actor, collection=collection,
        version=version, expected_revision=0)
    r = SourceRuntime(TurnBudget(TurnLimits()), authorization)
    original = "Explain A_12"
    with source_runtime_scope(r):
        await database_sync_to_async(ensure_admission)(r)
        vocab = await database_sync_to_async(_frozen_vocabulary)(r)
    rows = build_query_proposals(original, vocabulary=vocab, rewrite="Describe A_12")
    r._query_proposals = rows
    calls = []
    async def execute(action):
        assert r.budget.reserve_action(action.signature)
        calls.append(action.query)
        return {"result": []}
    async def assess(*args, **kwargs):
        raise AssertionError("fourth acquisition cannot plan")
    with source_runtime_scope(r):
        result = await acquire_evidence(original, budget=r.budget, llm=object(),
            execute=execute, evidence_views=lambda _: (), iterative=True,
            query_proposals=rows, assess=assess)
    assert calls == [row.text for row in rows]
    assert result.rounds == 3 and r.budget.planner_calls == 0


def test_proposal_cache_identity_binds_query_policy_and_resets(admitted_query_runtimes):
    r = admitted_query_runtimes[0]
    row = build_query_proposals("Original")[0]
    r._query_proposals = (row,)
    with source_runtime_scope(r):
        assert proposal_identity(r, row.text) == ("vocabulary-disabled-v1", None)
        with proposal_scope(row, runtime=r):
            first, bound = proposal_identity(r, row.text)
            assert bound.runtime is r
            with pytest.raises(ValueError):
                proposal_identity(r, "another")
            with pytest.raises(ValueError):
                with proposal_scope(replace(row, policy_key="other"), runtime=r):
                    pass
        assert proposal_identity(r, row.text) == ("vocabulary-disabled-v1", None)
        changed = replace(row, policy_key="other")
        r._query_proposals = (changed,)
        with proposal_scope(changed, runtime=r):
            assert proposal_identity(r, row.text)[0] != first


@pytest.mark.asyncio
async def test_concurrent_turn_proposal_contexts_do_not_share_identity(admitted_query_runtimes):
    async def one(text, r):
        row = build_query_proposals(text)[0]
        r._query_proposals = (row,)
        with source_runtime_scope(r), proposal_scope(row, runtime=r):
            await asyncio.sleep(0)
            return proposal_identity(r, text)[0]
    values = await asyncio.gather(one("first", admitted_query_runtimes[0]), one("second", admitted_query_runtimes[1]))
    assert len(set(values)) == 2


def test_unbound_synthetic_vocabulary_cannot_authorize_production():
    from apps.collections.services.vocabulary import validate_vocabulary
    snapshot = VocabularySnapshotV1((), (), (), "a" * 64)
    with pytest.raises(ValueError, match="unbound"):
        validate_vocabulary(snapshot, authorization=runtime().authorization)
