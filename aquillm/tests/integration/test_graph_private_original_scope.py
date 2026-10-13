"""Original private graph accounting cannot become public source-mode authority."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_lifetime import enter_search, graph_preflight
from apps.documents.services.graph_source_runtime import graph_source_runtime
from apps.documents.services.hybrid_graph_authorization import HybridGraphRetrievalDependencies
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import (
    SourceRuntime, current_source_runtime, source_runtime_scope, source_query_rows,
)
from apps.documents.tests.principal_scope_support import principals as principals
from lib.llm.turn_context import bind_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("change", [None, "parent", "authorization", "representation", "dense", "principal", "registry", "collections"])
def test_graph_runtime_reuses_exact_originals_without_refreeze(principals, monkeypatch, public, change):
    case, public_before = principals, None
    runtime = case.parent
    with bind_turn(case.turn), source_runtime_scope(runtime) if public else nullcontext():
        public_before = current_source_runtime()
        lifetime, scope = enter_search(TextChunk, (case.documents[0],), runtime.authorization, "Original", True)
        with lifetime:
            frozen, budget, col_ref = runtime._frozen_retrieval_scope, runtime.budget, case.consumer.col_ref
            query_identity = (scope.query_digest, scope.query_compatibility, scope.proposal_binding)
            monkeypatch.setattr("apps.documents.services.chunk_search_scope.freeze_active_representations",
                                lambda *a, **k: pytest.fail("graph may not recapture"))
            if change == "parent":
                del runtime._frozen_retrieval_scope
            elif change == "authorization":
                runtime.authorization = replace(runtime.authorization)
            elif change == "representation":
                runtime._frozen_retrieval_scope = replace(frozen, representation=replace(frozen.representation))
            elif change == "dense":
                runtime._frozen_retrieval_scope = replace(frozen, dense=object())
            elif change == "principal":
                case.consumer.user = case.same_pk_principal
            elif change == "registry":
                del case.turn._media_source_runtime
            elif change == "collections":
                case.consumer.col_ref = SimpleNamespace(collections=tuple(col_ref.collections))
            try:
                with CaptureQueriesContext(connection) as queries:
                    if change:
                        with pytest.raises(SourceScopeUnavailable):
                            graph_source_runtime()
                    else:
                        assert graph_source_runtime() is runtime
                assert len(queries) == 0
                if change is None:
                    rows = source_query_rows(TextChunk.objects.filter(pk=case.chunks[0].pk),
                                             runtime=graph_source_runtime(), require_existing=True)
                    assert rows[0].content == case.chunks[0].content
                    assert budget.sources_used == 1
                    assert runtime._frozen_retrieval_scope is frozen
                assert current_source_runtime() is public_before
                assert runtime.budget is budget and case.turn.budget is budget
                assert (scope.query_digest, scope.query_compatibility, scope.proposal_binding) == query_identity
            finally:
                runtime._frozen_retrieval_scope = frozen
                runtime.authorization = frozen.authorization
                case.consumer.user = case.alice
                case.turn._media_source_runtime = runtime
                case.consumer.col_ref = col_ref


def test_standalone_public_runtime_is_reused_and_no_context_is_not_authority():
    assert graph_source_runtime() is None
    # A standalone caller retains its existing public protocol; this is not a graph authority.
    runtime = SourceRuntime(TurnBudget(TurnLimits()), object())
    with source_runtime_scope(runtime):
        assert graph_source_runtime() is runtime
    assert current_source_runtime() is None


@pytest.mark.parametrize("mode", ["legacy", "shadow", "invalid"])
def test_nonnull_preflight_never_enters_non_v2_mode(mode):
    settings = SimpleNamespace(graph_remediation_mode=mode, memgraph_traversal_enabled=True,
                               graph_direct_enabled=True, graph_extended_enabled=False)
    runtime = SimpleNamespace(**{name: lambda **k: None for name in (
        "prepare_shared", "run_direct", "prepare_extended", "run_extended")})
    dependencies = HybridGraphRetrievalDependencies(runtime, settings, lambda **k: ())
    scope = SimpleNamespace(representation=SimpleNamespace(
        selections=(SimpleNamespace(generation_key=object()),)))
    assert graph_preflight(True, True, (), None, scope,
        lambda *a: pytest.fail("closed modes must not freeze"), dependencies=dependencies) == (None, None, "miss", ())
