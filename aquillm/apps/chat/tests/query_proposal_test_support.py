"""Authorized standalone query admissions established before async test work."""
from uuid import uuid4

import pytest

from apps.collections.tests.test_vocabulary_scope import setup_scope
from apps.chat.services.retrieval_query_binding import ensure_admission
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def admitted_query_runtimes(transactional_db):
    runtimes = []
    for _ in range(3):
        _, _, authorization = setup_scope("query-admission-" + uuid4().hex)
        runtime = SourceRuntime(TurnBudget(TurnLimits()), authorization)
        with source_runtime_scope(runtime):
            witness = ensure_admission(runtime)
        assert witness.scope is runtime._frozen_retrieval_scope
        runtimes.append(runtime)
    return runtimes


async def prepare_in_original_scope(*args, **kwargs):
    from apps.chat.services.retrieval_query_runtime import prepare_query_proposals
    runtime = kwargs["runtime"]
    original = runtime._frozen_retrieval_scope
    with source_runtime_scope(runtime):
        result = await prepare_query_proposals(*args, **kwargs)
    assert runtime._frozen_retrieval_scope is original
    return result
