from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def no_persisted_capacity_failure(monkeypatch):
    monkeypatch.setattr(_recovery(), "_exact_capacity_failure", lambda **_kwargs: False)


def _recovery():
    return import_module("apps.knowledge_graph.graph.recovery")




def test_exact_current_document_and_collection_artifacts_are_no_ops(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    monkeypatch.setattr(
        builds,
        "derive_current_document_build_key",
        lambda *_args: "b" * 64,
    )
    monkeypatch.setattr(
        builds,
        "_collection_context",
        lambda *_args: SimpleNamespace(identity=object()),
    )
    monkeypatch.setattr(
        builds, "derive_collection_build_key", lambda _identity: "c" * 64
    )
    monkeypatch.setattr(recovery, "_exact_artifact_exists", lambda **_kwargs: True)
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        Mock(side_effect=AssertionError("current document was republished")),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_collection_refresh",
        Mock(side_effect=AssertionError("current collection was republished")),
    )
    document_id = __import__("uuid").UUID("11111111-1111-4111-8111-111111111111")

    document = recovery._recover_document(document_id, "a" * 64)
    collection = recovery._recover_collection(17)

    assert document == recovery.RecoveryOutcome.CURRENT
    assert collection == recovery.RecoveryOutcome.CURRENT


def test_collection_with_missing_exact_document_dependency_is_deferred(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    monkeypatch.setattr(
        builds,
        "_collection_context",
        Mock(side_effect=builds.StaleBuildError("awaits a graph artifact")),
    )
    publish = Mock(side_effect=AssertionError("incomplete collection was published"))
    monkeypatch.setattr(builds, "enqueue_collection_refresh", publish)

    outcome = recovery._recover_collection(17)

    assert outcome == recovery.RecoveryOutcome.DEPENDENCY_PENDING
    publish.assert_not_called()


def test_recovery_page_is_bounded_and_returns_json_safe_continuation(monkeypatch):
    recovery = _recovery()
    rows = (
        recovery.DocumentRecoveryRow(
            document_id=__import__("uuid").UUID("11111111-1111-4111-8111-111111111111"),
            source_hash="a" * 64,
        ),
        recovery.DocumentRecoveryRow(
            document_id=__import__("uuid").UUID("22222222-2222-4222-8222-222222222222"),
            source_hash="b" * 64,
        ),
    )
    next_cursor = recovery.GraphRecoveryCursor(
        phase="documents", document_model_index=0, last_pk=22
    )
    monkeypatch.setattr(
        recovery,
        "_load_document_page",
        lambda cursor, page_size: (rows, next_cursor),
    )
    outcomes = iter(
        (recovery.RecoveryOutcome.CURRENT, recovery.RecoveryOutcome.PUBLISHED)
    )
    monkeypatch.setattr(
        recovery,
        "_recover_document",
        lambda *_args: next(outcomes),
    )

    summary = recovery.recover_graph_builds_page(None, page_size=2)

    assert summary == {
        "examined_count": 2,
        "current_count": 1,
        "published_count": 1,
        "dependency_pending_count": 0,
        "invalid_count": 0,
        "publish_failed_count": 0,
        "capacity_blocked_count": 0,
        "next_cursor": {
            "phase": "documents",
            "document_model_index": 0,
            "last_pk": 22,
        },
    }


def test_malformed_legacy_rows_do_not_block_valid_following_document(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    malformed_id = __import__("uuid").UUID("11111111-1111-4111-8111-111111111111")
    valid_id = __import__("uuid").UUID("22222222-2222-4222-8222-222222222222")
    rows = (
        recovery.DocumentRecoveryRow(
            document_id=malformed_id,
            source_hash="legacy-not-a-sha256",
        ),
        recovery.DocumentRecoveryRow(
            document_id=__import__("uuid").UUID(int=0),
            source_hash="a" * 64,
        ),
        recovery.DocumentRecoveryRow(document_id=valid_id, source_hash="a" * 64),
    )
    next_cursor = recovery.GraphRecoveryCursor(
        phase="documents", document_model_index=0, last_pk=22
    )
    monkeypatch.setattr(
        recovery,
        "_load_document_page",
        lambda cursor, page_size: (rows, next_cursor),
    )
    monkeypatch.setattr(
        builds,
        "derive_current_document_build_key",
        lambda *_args: "b" * 64,
    )
    monkeypatch.setattr(recovery, "_exact_artifact_exists", lambda **_kwargs: False)
    publish = Mock()
    monkeypatch.setattr(builds, "enqueue_document_build", publish)

    summary = recovery.recover_graph_builds_page(None, page_size=3)

    assert summary["examined_count"] == 3
    assert summary["invalid_count"] == 2
    assert summary["published_count"] == 1
    assert summary["next_cursor"]["last_pk"] == 22
    publish.assert_called_once_with(valid_id, "a" * 64)


def test_corrupt_collection_does_not_block_valid_following_collection(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    cursor = recovery.GraphRecoveryCursor(phase="collections", last_pk=0)
    next_cursor = recovery.GraphRecoveryCursor(phase="collections", last_pk=18)
    monkeypatch.setattr(
        recovery,
        "_load_collection_page",
        lambda _cursor, _page_size: ((17, 18), next_cursor),
    )

    def context(collection_id):
        if collection_id == 17:
            raise builds.CorruptBuildError("private persisted row detail")
        return SimpleNamespace(
            identity=SimpleNamespace(aggregate_source_signature="a" * 64)
        )

    monkeypatch.setattr(builds, "_collection_context", context)
    monkeypatch.setattr(
        builds, "derive_collection_build_key", lambda _identity: "c" * 64
    )
    monkeypatch.setattr(recovery, "_exact_artifact_exists", lambda **_kwargs: False)
    publish = Mock()
    monkeypatch.setattr(builds, "enqueue_collection_refresh", publish)

    summary = recovery.recover_graph_builds_page(cursor.as_dict(), page_size=2)

    assert summary["examined_count"] == 2
    assert summary["invalid_count"] == 1
    assert summary["published_count"] == 1
    assert summary["next_cursor"]["last_pk"] == 18
    publish.assert_called_once_with(18, "a" * 64, "c" * 64)


def test_document_source_limit_error_is_counted_as_invalid(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    pipeline = import_module("apps.knowledge_graph.extraction.pipeline")
    document_id = __import__("uuid").UUID("11111111-1111-4111-8111-111111111111")
    monkeypatch.setattr(
        builds,
        "derive_current_document_build_key",
        Mock(side_effect=pipeline.StaleSourceError("private source cap detail")),
    )

    outcome = recovery._recover_document(document_id, "a" * 64)

    assert outcome == recovery.RecoveryOutcome.INVALID


@pytest.mark.parametrize("kind", ["document", "collection"])
@pytest.mark.parametrize("stage", ["preflight", "publication"])
def test_capacity_blocked_row_does_not_starve_following_work(monkeypatch, kind, stage):
    from uuid import UUID

    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    pipeline = import_module("apps.knowledge_graph.extraction.pipeline")
    ids = (
        (
            UUID("11111111-1111-4111-8111-111111111111"),
            UUID("22222222-2222-4222-8222-222222222222"),
        )
        if kind == "document"
        else (17, 18)
    )
    published = []

    def check(identifier, at):
        if identifier == ids[0] and stage == at:
            raise pipeline.ExtractionCapacityError(
                pipeline.ExtractionCapacityCode.CHUNK_LIMIT,
                "private-source-canary",
            )

    def publish(identifier, *args):
        check(identifier, "publication")
        published.append(identifier)

    monkeypatch.setattr(recovery, "_exact_artifact_exists", lambda **kwargs: False)
    if kind == "document":

        def document_key(identifier, source_hash):
            check(identifier, "preflight")
            return "b" * 64

        monkeypatch.setattr(builds, "derive_current_document_build_key", document_key)
        monkeypatch.setattr(builds, "enqueue_document_build", publish)
        cursor = recovery.GraphRecoveryCursor(phase="documents")
        next_cursor = recovery.GraphRecoveryCursor(phase="documents", last_pk=22)
        rows = tuple(recovery.DocumentRecoveryRow(value, "a" * 64) for value in ids)
        monkeypatch.setattr(
            recovery, "_load_document_page", lambda *_args: (rows, next_cursor)
        )
    else:

        def collection_context(identifier):
            check(identifier, "preflight")
            return SimpleNamespace(
                identity=SimpleNamespace(aggregate_source_signature="a" * 64)
            )

        monkeypatch.setattr(builds, "_collection_context", collection_context)
        monkeypatch.setattr(builds, "derive_collection_build_key", lambda _: "b" * 64)
        monkeypatch.setattr(builds, "enqueue_collection_refresh", publish)
        cursor = recovery.GraphRecoveryCursor(phase="collections")
        next_cursor = recovery.GraphRecoveryCursor(phase="collections", last_pk=22)
        monkeypatch.setattr(
            recovery, "_load_collection_page", lambda *_args: (ids, next_cursor)
        )

    summary = recovery.recover_graph_builds_page(cursor.as_dict(), page_size=2)

    assert summary["capacity_blocked_count"] == 1
    assert summary["published_count"] == 1
    assert summary["publish_failed_count"] == 0
    assert summary["examined_count"] == 2
    assert summary["next_cursor"]["last_pk"] == 22
    assert published == [ids[1]]
    assert "private-source-canary" not in str(summary)
