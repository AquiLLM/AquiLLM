"""An expired native graph child cannot spend its reserved tail on deferred bodies."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import pytest
from django.conf import settings

from apps.documents.services.chunk_search_scope import bind_search_scope, reset_search_scope
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(autouse=True)
def full_candidate_limit(monkeypatch):
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 3.0, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)


def install_failure(monkeypatch, active, now, *, expired, mutation=None, early_timeout=False):
    from apps.documents.services import source_body_batch
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as ready
    from lib import replay_observation

    f, _, source, scope, _, _, operation, _ = active
    original, fetch = ready.select_ready_scope_v2, source_body_batch._fetch_body_rows
    tokens, fetched, events = [], [], []

    def select(**kwargs):
        result = original(**kwargs)  # Real fresh authority before the failure.
        if expired:
            now[0] = 6.0  # Graph child cap5s; original action10s remains live.
        if mutation == "operation":
            operation.live = False
        elif mutation == "permission":
            f.policy.rows = ()
        elif mutation == "context":
            tokens.append(bind_search_scope(replace(scope), "Atlas", graph_enabled=True))
        elif mutation == "cleanup":
            database = scope.database

            @contextmanager
            def revoke_after_cleanup():
                with database():
                    yield
                operation.live = False

            monkeypatch.setattr(scope, "database", revoke_after_cleanup)
        elif mutation == "source":
            def committed():
                from django.db import connection

                try:
                    f.documents["A"].full_text += " changed"
                    f.documents["A"].full_text_hash = sha256(f.documents["A"].full_text.encode()).hexdigest()
                    f.documents["A"].save(dont_rechunk=True)
                finally:
                    connection.close()

            # The native read transaction rolls back when its child expires;
            # this source race must remain committed for fallback to observe it.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(committed).result(timeout=10)
        if not expired:
            if early_timeout:
                raise TimeoutError("synthetic provider timeout before phase expiry")
            raise ValueError("synthetic early graph failure")
        return result

    def read(*args, **kwargs):
        rows = fetch(*args, **kwargs)
        fetched.extend(row.pk for row in rows)
        return rows

    monkeypatch.setattr(ready, "select_ready_scope_v2", select)
    monkeypatch.setattr(source_body_batch, "_fetch_body_rows", read)
    monkeypatch.setattr(replay_observation, "graph_selected_sources_v2", events.append)
    return tokens, fetched, events


@pytest.mark.parametrize("mode", ["expired", "early", "early_timeout"])
def test_actual_native_expired_child_keeps_initial_bodies_early_failure_still_tops_up(monkeypatch, mode):
    expired = mode == "expired"
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True,
                     budget=budget, timeout_ms=10_000) as active:
        f, graph, source, scope, _, dependencies, operation, _ = active
        graph.clock = lambda: now[0]
        snapshot = capture(active)
        assert len(snapshot.baseline_candidates) == source.budget.sources_used == 25
        before = source.budget.text_used.copy()
        _, fetched, events = install_failure(monkeypatch, active, now, expired=expired,
                                           early_timeout=mode == "early_timeout")
        rows, diagnostic = hybrid_graph_candidate_pool_v2(snapshot, "Atlas", f.authorization, dependencies)
        assert operation.active() and budget.can_publish()
        assert diagnostic["graph_status"] == "error" and events == []
        assert diagnostic["source_admission"]["fallback_top_up_skipped"] is expired
        assert diagnostic["source_admission"]["fallback_top_up_count"] == (0 if expired else 20)
        assert len(rows) == budget.sources_used == (25 if expired else 45)
        assert set(scope.bodies) == {r.pk for r in rows}
        if expired:
            assert tuple(r.pk for r in rows) == tuple(r.pk for r in snapshot.baseline_candidates)
            assert fetched == [] and budget.text_used == before
        else:
            assert len(fetched) == 20 and budget.text_used["materialized"] > before["materialized"]


@pytest.mark.parametrize("mutation", ["operation", "permission", "context", "source", "cleanup"])
def test_expired_child_fallback_cannot_replace_or_retain_lost_original_authority(monkeypatch, mutation):
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True,
                     budget=budget, timeout_ms=10_000) as active:
        f, graph, source, scope, _, dependencies, _, _ = active
        graph.clock = lambda: now[0]
        snapshot = capture(active)
        before = (source.cache.copy(), scope.bodies.copy(), budget.text_used.copy())
        tokens, fetched, events = install_failure(monkeypatch, active, now, expired=True, mutation=mutation)
        try:
            rows, diagnostic = hybrid_graph_candidate_pool_v2(snapshot, "Atlas", f.authorization, dependencies)
            assert rows == () and diagnostic["graph_status"] == "error"
            assert fetched == [] and events == [] and budget.sources_used == 25
            assert source.cache == before[0] and scope.bodies == before[1]
            assert budget.text_used == before[2]
        finally:
            for token in reversed(tokens):
                reset_search_scope(token)
