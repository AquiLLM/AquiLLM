from __future__ import annotations

import uuid


SOURCE_HASH = "a" * 64


def test_current_collection_refresh_is_a_public_build_service_seam():
    from apps.knowledge_graph.services import builds

    assert "enqueue_current_collection_refresh" in builds.__all__
    assert callable(builds.enqueue_current_collection_refresh)


def test_post_chunk_enqueue_waits_for_commit_and_uses_an_exact_uuid(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks = []
    robust_flags = []
    database_aliases = []
    queued = []

    def capture(callback, *, using=None, robust=False):
        callbacks.append(callback)
        robust_flags.append(robust)
        database_aliases.append(using)

    monkeypatch.setattr(invalidation.transaction, "on_commit", capture)
    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda document_id, source_hash: queued.append((document_id, source_hash)),
    )

    document_id = uuid.uuid4()
    invalidation.schedule_post_chunk_graph_build(
        document_id, SOURCE_HASH, using="default"
    )

    assert queued == []
    assert len(callbacks) == 1
    assert robust_flags == [True]
    assert database_aliases == ["default"]

    callbacks[0]()

    assert queued == [(document_id, SOURCE_HASH)]
    assert type(queued[0][0]) is uuid.UUID


def test_post_chunk_queue_failure_is_logged_without_escaping(monkeypatch):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks = []
    events = []

    monkeypatch.setattr(
        invalidation.transaction,
        "on_commit",
        lambda callback, **_kwargs: callbacks.append(callback),
    )

    monkeypatch.setattr(
        builds,
        "enqueue_document_build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("broker secret")),
    )
    monkeypatch.setattr(
        invalidation.logger,
        "error",
        lambda event, **fields: events.append((event, fields)),
    )

    document_id = uuid.uuid4()
    invalidation.schedule_post_chunk_graph_build(document_id, SOURCE_HASH)
    callbacks[0]()

    assert events == [
        (
            "obs.kg.document_enqueue_failed",
            {
                "document_id": str(document_id),
                "expected_source_hash": SOURCE_HASH,
                "error_type": "ConnectionError",
            },
        )
    ]
