from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from django.db import close_old_connections, connection
from django.utils import timezone

from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
    _document_context,
    _persist_active_ontology,
    _persist_document,
    database_required,
)

pytestmark = [pytest.mark.django_db(transaction=True), database_required]


def _backend_pid() -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        row = cursor.fetchone()
    assert row is not None
    return int(row[0])


def _wait_until_blocked_by(
    blocked_pid: int,
    blocker_pid: int,
    *,
    timeout: float = 10,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT wait_event_type, pg_blocking_pids(pid)
                FROM pg_stat_activity
                WHERE pid = %s
                """,
                [blocked_pid],
            )
            row = cursor.fetchone()
        if row is not None and row[0] == "Lock" and blocker_pid in row[1]:
            return
        time.sleep(0.01)
    raise AssertionError("request advance did not block on the worker transition")


def test_bootstrapped_worker_transition_and_request_advance_do_not_deadlock(
    monkeypatch,
):
    from apps.knowledge_graph.models import GraphBuildRun, GraphRebuildRequest
    from apps.knowledge_graph.services import builds
    from lib.knowledge_graph import config

    _persist_active_ontology()
    collection, document, chunk = _persist_document(label="transition-advance")
    context = _document_context(document, chunk)
    request = GraphRebuildRequest.objects.create(
        id=uuid.uuid4(),
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=str(document.id),
        requested_documents=[
            {
                "document_id": str(document.id),
                "document_pkid": document.pkid,
                "model_label": document._meta.label_lower,
                "collection_id": collection.pk,
                "source_hash": document.full_text_hash,
            }
        ],
        document_count=1,
        document_publish_cursor=1,
        document_publication_state=(GraphRebuildRequest.PublicationState.PUBLISHED),
        status=GraphRebuildRequest.Status.RUNNING,
        started_at=timezone.now(),
    )
    monkeypatch.setattr(builds, "_document_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(config, "load_extraction_settings", lambda: context.settings)
    build_key = builds.derive_document_build_key(context.identity)
    artifact, run, owner, lease_generation, completed = (
        builds._bootstrap_document_build(
            context,
            build_key,
            request.pk,
            False,
        )
    )
    assert completed is False
    assert owner is not None and lease_generation is not None

    worker_updated = Event()
    release_worker = Event()
    manager_connected = Event()
    pids: dict[str, int] = {}
    real_renew = builds.renew_build_lease

    def pause_after_renew(run_id, lease_owner, generation):
        real_renew(run_id, lease_owner, generation)
        worker_updated.set()
        assert release_worker.wait(timeout=20)

    monkeypatch.setattr(builds, "renew_build_lease", pause_after_renew)

    def transition_run():
        close_old_connections()
        try:
            pids["worker"] = _backend_pid()
            return builds._transition_run(
                run.pk,
                GraphBuildRun.Stage.EXTRACTING,
                lease_owner=owner,
                lease_generation=lease_generation,
            )
        finally:
            close_old_connections()

    def advance_request():
        close_old_connections()
        try:
            pids["manager"] = _backend_pid()
            manager_connected.set()
            builds.advance_rebuild_request(request.pk)
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        transition = executor.submit(transition_run)
        assert worker_updated.wait(timeout=10)
        advance = executor.submit(advance_request)
        assert manager_connected.wait(timeout=10)
        try:
            _wait_until_blocked_by(pids["manager"], pids["worker"])
        finally:
            release_worker.set()
        transitioned = transition.result(timeout=20)
        advance.result(timeout=20)

    run.refresh_from_db()
    request.refresh_from_db()
    assert transitioned.stage == run.stage == GraphBuildRun.Stage.EXTRACTING
    assert request.status == GraphRebuildRequest.Status.RUNNING
    assert request.completed_document_count == 0


def test_worker_transition_commit_and_failure_recording_do_not_deadlock(monkeypatch):
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.services import builds
    from lib.knowledge_graph import config

    _persist_active_ontology()
    collection, document, chunk = _persist_document(label="completion-failure")
    context = _document_context(document, chunk)
    monkeypatch.setattr(builds, "_document_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(config, "load_extraction_settings", lambda: context.settings)
    request = GraphRebuildRequest.objects.create(
        id=uuid.uuid4(),
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(collection.pk),
        requested_documents=[
            {
                "document_id": str(document.id),
                "document_pkid": document.pkid,
                "model_label": document._meta.label_lower,
                "collection_id": collection.pk,
                "source_hash": document.full_text_hash,
            }
        ],
        document_count=1,
        collection_count=1,
        status=GraphRebuildRequest.Status.RUNNING,
        evaluation_only=True,
        started_at=timezone.now(),
    )
    build_key = builds.derive_document_build_key(context.identity)
    artifact, run, owner, lease_generation, completed = (
        builds._bootstrap_document_build(
            context,
            build_key,
            request.pk,
            True,
        )
    )
    assert completed is False
    assert owner is not None and lease_generation is not None
    worker_updated = Event()
    release_worker = Event()
    manager_connected = Event()
    pids: dict[str, int] = {}
    queries: list[str] = []
    real_renew = builds.renew_build_lease

    def pause_after_renew(run_id, lease_owner, generation):
        real_renew(run_id, lease_owner, generation)
        worker_updated.set()
        assert release_worker.wait(timeout=20)

    monkeypatch.setattr(builds, "renew_build_lease", pause_after_renew)

    def transition_run():
        close_old_connections()
        try:
            pids["worker"] = _backend_pid()
            return builds._transition_run(
                run.pk,
                GraphBuildRun.Stage.EXTRACTING,
                lease_owner=owner,
                lease_generation=lease_generation,
            )
        finally:
            close_old_connections()

    def record_failure():
        close_old_connections()
        try:
            pids["manager"] = _backend_pid()
            manager_connected.set()

            def capture(execute, sql, params, many, context):
                queries.append(sql.upper())
                return execute(sql, params, many, context)

            with connection.execute_wrapper(capture):
                builds.record_rebuild_failure(
                    request.pk,
                    error_code="collection_rebuild_failed",
                )
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        transitioning = executor.submit(transition_run)
        assert worker_updated.wait(timeout=10)
        recording = executor.submit(record_failure)
        assert manager_connected.wait(timeout=10)
        try:
            try:
                _wait_until_blocked_by(pids["manager"], pids["worker"])
            except AssertionError:
                recording.result(timeout=1)
                raise
        finally:
            release_worker.set()
        transitioned = transitioning.result(timeout=20)
        recording.result(timeout=20)

    request.refresh_from_db()
    assert transitioned.stage == GraphBuildRun.Stage.EXTRACTING
    assert request.status == GraphRebuildRequest.Status.FAILED
    assert request.completed_document_count == 0
    assert request.terminal_failure_count == 1
    assert request.failed_collection_count == 1
    artifact_table = GraphArtifact._meta.db_table.upper()
    run_table = GraphBuildRun._meta.db_table.upper()
    artifact_lock = next(
        sql for sql in queries if artifact_table in sql and "FOR " in sql
    )
    run_lock = next(sql for sql in queries if run_table in sql and "FOR " in sql)
    assert "FOR NO KEY UPDATE" in artifact_lock
    assert "FOR UPDATE" in run_lock
    assert queries.index(artifact_lock) < queries.index(run_lock)
