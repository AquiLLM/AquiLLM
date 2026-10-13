"""Original-budget fences across authorization, SQL, setup and dispatch."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import monotonic
from unittest.mock import patch

import pytest
from django.db import connection, connections, transaction, OperationalError
from django.utils import timezone

from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_store as store
from apps.documents.services import embedding_index_query_scope as query_scope
from apps.documents.services import representation_store as representations
from apps.documents.services.embedding_index_deadline import embedding_build_transaction
from apps.documents.tasks import embedding_index as task
from apps.documents.tests.test_embedding_generation import legacy as legacy, stage
from apps.documents.tests.test_embedding_generation_query import (
    runtime_for,
    rank,
    query,
)
from apps.documents.services.source_loading import SourcePreparationLimited
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.document_provider import observe_query_embedding
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def schedule(legacy, transport):
    space, _ = transport
    store.persist_embedding_space(space, using="default")
    selection = representations.freeze_active_representations(
        (legacy[1],), using="default"
    ).selections[0]
    with patch.object(task.build_embedding_generation, "delay"):
        key = task.rebuild_document_embeddings(
            selection, space_key=space.space_key, using="default"
        )
    return selection, key


def test_final_reauthorization_cannot_return_after_original_turn_deadline(
    legacy, transport, monkeypatch
):
    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    clock = [0.0]
    runtime, _ = runtime_for(legacy)
    runtime = replace(runtime, budget=TurnBudget(TurnLimits(), clock=lambda: clock[0]))
    observed = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    original = query_scope.revalidate_retrieval_authorization_context
    calls = []

    def reauthorize(**kwargs):
        result = original(**kwargs)
        calls.append(True)
        if len(calls) == 2:
            clock[0] = 15.01
        return result

    monkeypatch.setattr(
        query_scope, "revalidate_retrieval_authorization_context", reauthorize
    )
    with pytest.raises(SourcePreparationLimited, match="source deadline"):
        rank(frozen, dense, observed, runtime)
    assert len(calls) == 2


def test_build_sql_preserves_stricter_operator_timeout():
    with connection.cursor() as cursor:
        cursor.execute("SET statement_timeout = '250ms'")
    try:
        with embedding_build_transaction(build_budget(), using="default"):
            with connection.cursor() as cursor:
                cursor.execute("SHOW statement_timeout")
                assert cursor.fetchone()[0] == "250ms"
        with connection.cursor() as cursor:
            cursor.execute("SHOW statement_timeout")
            assert cursor.fetchone()[0] == "250ms"
    finally:
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = 0")


@pytest.mark.parametrize("deferred", [False, True])
def test_schedule_does_not_enqueue_after_original_deadline(
    legacy, transport, monkeypatch, deferred
):
    selection, _ = schedule(legacy, transport)
    clock = [0.0]
    budget = build_budget(clock=lambda: clock[0])
    monkeypatch.setattr(task, "build_budget", lambda timeout_ms: budget)
    original = task.begin_embedding_generation

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        if not deferred:
            clock[0] = 121.0
        return result

    monkeypatch.setattr(task, "begin_embedding_generation", slow)
    with patch.object(task.build_embedding_generation, "delay") as queue:
        with pytest.raises(TimeoutError, match="embedding_deadline"):
            with transaction.atomic():
                task.rebuild_document_embeddings(
                    selection, space_key=transport[0].space_key, using="default"
                )
                if deferred:
                    clock[0] = 121.0
        queue.assert_not_called()


def test_worker_lease_lock_wait_is_bounded_by_original_deadline(legacy, transport):
    _, key = schedule(legacy, transport)
    locked, release = Event(), Event()

    def hold():
        try:
            with transaction.atomic():
                EmbeddingGeneration.objects.select_for_update().get(pk=key)
                locked.set()
                release.wait(3)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold)
        assert locked.wait(2)
        started = monotonic()
        try:
            with pytest.raises((TimeoutError, OperationalError)):
                task.build_embedding_generation.run(key, timeout_ms=200)
            assert monotonic() - started < 1.0
        finally:
            release.set()
            future.result()
    row = EmbeddingGeneration.objects.get(pk=key)
    assert row.deliveries == 0
    with connection.cursor() as cursor:
        cursor.execute("SHOW statement_timeout")
        assert cursor.fetchone()[0] == "0"


@pytest.mark.parametrize("table", ["embeddingspace", "chunkembedding", "textchunk"])
def test_worker_stops_setup_and_preparation_after_read_expiry(
    legacy, transport, monkeypatch, table
):
    _, key = schedule(legacy, transport)
    clock = [0.0]
    budget = build_budget(clock=lambda: clock[0])
    monkeypatch.setattr(task, "build_budget", lambda timeout_ms: budget)
    late_sql = []

    def expire(execute, sql, params, many, context):
        expired_before = bool(clock[0])
        result = execute(sql, params, many, context)
        if expired_before and ("apps_documents_" in sql or "aquillm_textchunk" in sql):
            late_sql.append(sql)
        target = (
            "aquillm_textchunk" if table == "textchunk" else f"apps_documents_{table}"
        )
        if sql.startswith("SELECT") and f'"{target}"' in sql:
            clock[0] = 121.0
        return result

    original = task.prepare_document

    def prepare(*args, **kwargs):
        assert clock[0] == 0.0, "preparation started after expiry"
        return original(*args, **kwargs)

    monkeypatch.setattr(task, "prepare_document", prepare)
    with (
        connection.execute_wrapper(expire),
        patch.object(task, "embed_document_batch") as provider,
    ):
        with pytest.raises(TimeoutError, match="embedding_deadline"):
            task.build_embedding_generation.run(key)
        provider.assert_not_called()
    assert not late_sql


def test_expired_claims_require_explicit_rebuild_and_exhaust_honestly(
    legacy, transport, monkeypatch
):
    selection, key = schedule(legacy, transport)
    real_budget = task.build_budget
    original = task.load_embedding_space
    for attempt in range(1, 4):
        clock = [0.0]
        budget = build_budget(clock=lambda: clock[0])
        monkeypatch.setattr(task, "build_budget", lambda timeout_ms: budget)

        def expired(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 121.0
            return result

        monkeypatch.setattr(task, "load_embedding_space", expired)
        with patch.object(task.build_embedding_generation, "retry") as retry:
            with pytest.raises(TimeoutError):
                task.build_embedding_generation.run(key)
            retry.assert_not_called()
        row = EmbeddingGeneration.objects.get(pk=key)
        assert (row.deliveries, row.state, row.failure_kind) == (
            attempt,
            "building",
            "",
        )
        monkeypatch.setattr(task, "build_budget", real_budget)
        monkeypatch.setattr(task, "load_embedding_space", original)
        EmbeddingGeneration.objects.filter(pk=key).update(
            next_attempt_at=timezone.now()
        )
        with patch.object(task.build_embedding_generation, "delay") as queue:
            assert (
                task.rebuild_document_embeddings(
                    selection, space_key=transport[0].space_key, using="default"
                )
                == key
            )
            queue.assert_called_once_with(key, using="default", timeout_ms=120000)
    assert task.build_embedding_generation.run(key) == "not_claimed"
    row.refresh_from_db()
    assert (row.deliveries, row.state, row.failure_kind) == (
        3,
        "failed",
        "delivery_exhausted",
    )


def test_deadline_database_scope_holds_no_media_preparation_locks(
    legacy, transport, monkeypatch
):
    from apps.documents.services import embedding_index_manifest as manifest

    original = manifest.prepare_document
    calls = []

    def prepare(*args, **kwargs):
        assert not connection.in_atomic_block
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(manifest, "prepare_document", prepare)
    monkeypatch.setattr(task, "prepare_document", prepare)
    _, key = schedule(legacy, transport)
    assert task.build_embedding_generation.run(key) == "activated"
    assert len(calls) >= 3


@pytest.mark.parametrize("expired", [False, True])
def test_upstream_retry_never_restarts_expired_budget(legacy, transport, expired):
    from celery.exceptions import Retry
    from lib.embeddings.errors import EmbeddingUpstreamUnavailableError

    _, key = schedule(legacy, transport)
    clock = [0.0]
    budget = build_budget(clock=lambda: clock[0])

    def unavailable(*args, **kwargs):
        assert kwargs["budget"] is budget
        clock[0] = 121.0 if expired else 5.0
        raise EmbeddingUpstreamUnavailableError("synthetic outage")

    with (
        patch.object(task, "build_budget", return_value=budget) as factory,
        patch.object(task, "embed_document_batch", side_effect=unavailable),
        patch.object(
            task.build_embedding_generation, "retry", side_effect=Retry()
        ) as retry,
    ):
        with pytest.raises(TimeoutError if expired else Retry):
            task.build_embedding_generation.run(key)
        factory.assert_called_once_with(120000)
        if expired:
            retry.assert_not_called()
        else:
            retry.assert_called_once_with(countdown=30)
    row = EmbeddingGeneration.objects.get(pk=key)
    assert (row.deliveries, row.state, row.failure_kind) == (1, "building", "")
