from __future__ import annotations

import uuid

import pytest


@pytest.mark.parametrize(
    ("ingestion_complete", "has_active_document_artifact"),
    ((False, True), (True, False)),
)
def test_move_withholds_current_collection_until_chunks_and_document_graph_are_ready(
    monkeypatch,
    ingestion_complete,
    has_active_document_artifact,
):
    from apps.knowledge_graph.graph import invalidation
    from apps.knowledge_graph.services import builds

    callbacks = []
    refreshed = []
    monkeypatch.setattr(
        invalidation.transaction,
        "on_commit",
        lambda callback, **_kwargs: callbacks.append(callback),
    )
    monkeypatch.setattr(
        invalidation,
        "cleanup_document_collection_graph_state",
        lambda *_args, **_kwargs: invalidation.DocumentGraphCleanupResult(
            affected_collection_ids=(17, 19),
            current_collection_id=19,
            source_hash="a" * 64,
            ingestion_complete=ingestion_complete,
            has_active_document_artifact=has_active_document_artifact,
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "load_current_document_lifecycle_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routing must use the immutable locked cleanup snapshot")
        ),
    )
    monkeypatch.setattr(
        builds,
        "enqueue_current_collection_refresh",
        refreshed.append,
    )
    event = invalidation.DocumentLifecycleEvent(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=41,
            document_id=uuid.uuid4(),
        ),
        old_source_hash="a" * 64,
        committed_source_hash="a" * 64,
        old_collection_id=17,
        committed_collection_id=19,
    )

    invalidation.schedule_document_move_invalidation(event)
    callbacks[0]()

    assert refreshed == [17]
