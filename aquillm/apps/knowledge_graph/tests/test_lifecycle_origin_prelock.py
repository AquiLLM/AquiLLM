from __future__ import annotations

import uuid
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


def _document_ref(invalidation, *, pkid=41, collection_id=17):
    return invalidation.DeleteDocumentSnapshot(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=pkid,
            document_id=uuid.uuid4(),
        ),
        source_hash="a" * 64,
        collection_id=collection_id,
        figure_parent_identity=None,
    )


@pytest.mark.parametrize("receiver_name", ("document_pre_delete", "collection_pre_delete"))
def test_source_delete_receivers_fail_closed_without_an_origin(receiver_name):
    from apps.knowledge_graph.graph import invalidation

    receiver = getattr(invalidation, receiver_name)
    sender = SimpleNamespace(
        _meta=SimpleNamespace(label_lower="apps_documents.rawtextdocument")
    )
    instance = SimpleNamespace(
        pk=17,
        pkid=41,
        id=uuid.uuid4(),
        parent_id=None,
        collection_id=17,
        full_text_hash="a" * 64,
    )

    with pytest.raises(RuntimeError, match="origin"):
        receiver(sender, instance, using="default", origin=None)


def test_origin_prelock_acquires_the_complete_sorted_scope_once(monkeypatch):
    from apps.knowledge_graph.graph import invalidation

    document = _document_ref(invalidation)
    owner_fence = _document_ref(invalidation, pkid=43, collection_id=19)
    snapshot = invalidation.DeleteScopeSnapshot(
        collection_rows=((17, None), (19, 17)),
        documents=(document,),
        fence_documents=(document, owner_fence),
        locked_collection_ids=(17, 19, 23),
    )
    snapshots = [snapshot, snapshot]
    actions = []
    origin = SimpleNamespace()
    token = object()
    collection_graph = invalidation.GraphRowLockSet(
        artifact_ids=(101, 103),
        run_ids=(201, 203),
    )
    document_graph = invalidation.GraphRowLockSet(
        artifact_ids=(107,),
        run_ids=(207,),
    )
    monkeypatch.setattr(
        invalidation,
        "_delete_transaction_token",
        lambda *, using: token,
    )
    monkeypatch.setattr(
        invalidation,
        "_snapshot_origin_delete_scope",
        lambda _origin, *, using: snapshots.pop(0),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_scopes",
        lambda collection_ids, *, using: actions.append(
            ("collection_advisory", collection_ids, using)
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_rows",
        lambda collection_ids, *, using: actions.append(
            ("collection_rows", collection_ids, using)
        )
        or ((17, None), (19, 17), (23, None)),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_graph_rows",
        lambda collection_ids, *, using: actions.append(
            ("collection_graph", collection_ids, using)
        )
        or collection_graph,
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_delete_document_scopes",
        lambda documents, *, using: actions.append(
            ("document_advisory", documents, using)
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_document_graph_rows",
        lambda documents, *, using: actions.append(
            ("document_graph", documents, using)
        )
        or document_graph,
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_delete_document_rows",
        lambda documents, *, using: actions.append(
            ("document_rows", documents, using)
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_discover_collection_graph_rows",
        lambda *_args, **_kwargs: collection_graph,
    )
    monkeypatch.setattr(
        invalidation,
        "_discover_document_graph_rows",
        lambda *_args, **_kwargs: document_graph,
    )

    context = invalidation._ensure_origin_delete_scope_locked(
        origin,
        using="default",
    )
    repeated = invalidation._ensure_origin_delete_scope_locked(
        origin,
        using="default",
    )

    assert repeated is context
    assert context.snapshot is snapshot
    assert actions == [
        ("collection_advisory", (17, 19, 23), "default"),
        ("collection_rows", (17, 19, 23), "default"),
        ("collection_graph", (17, 19, 23), "default"),
        ("document_advisory", (document, owner_fence), "default"),
        (
            "document_graph",
            (document.document, owner_fence.document),
            "default",
        ),
        ("document_rows", (document, owner_fence), "default"),
    ]
    assert context.collection_graph_rows == collection_graph
    assert context.document_graph_rows == document_graph
    with pytest.raises(FrozenInstanceError):
        context.using = "archive"


def test_origin_prelock_fails_closed_when_scope_expands_under_locks(monkeypatch):
    from apps.knowledge_graph.graph import invalidation

    document = _document_ref(invalidation)
    first = invalidation.DeleteScopeSnapshot(
        collection_rows=((17, None),),
        documents=(document,),
        locked_collection_ids=(17,),
    )
    expanded = invalidation.DeleteScopeSnapshot(
        collection_rows=((17, None),),
        documents=(document,),
        locked_collection_ids=(17, 19),
    )
    snapshots = [first, expanded]
    monkeypatch.setattr(
        invalidation,
        "_delete_transaction_token",
        lambda *, using: object(),
    )
    monkeypatch.setattr(
        invalidation,
        "_snapshot_origin_delete_scope",
        lambda _origin, *, using: snapshots.pop(0),
    )
    monkeypatch.setattr(invalidation, "_lock_collection_scopes", lambda *_a, **_k: None)
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_rows",
        lambda *_a, **_k: ((17, None),),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_graph_rows",
        lambda *_a, **_k: invalidation.GraphRowLockSet(),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_delete_document_scopes",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_document_graph_rows",
        lambda *_a, **_k: invalidation.GraphRowLockSet(),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_delete_document_rows",
        lambda *_a, **_k: None,
    )

    with pytest.raises(RuntimeError, match="changed while lifecycle locks"):
        invalidation._ensure_origin_delete_scope_locked(
            SimpleNamespace(),
            using="default",
        )


def test_stale_document_receiver_must_belong_to_the_locked_origin_scope():
    from apps.knowledge_graph.graph import invalidation

    locked = _document_ref(invalidation, pkid=41)
    context = invalidation.OriginDeleteContext(
        using="default",
        snapshot=invalidation.DeleteScopeSnapshot(
            collection_rows=((17, None),),
            documents=(locked,),
            locked_collection_ids=(17,),
        ),
    )
    stale = invalidation.DeleteDocumentSnapshot(
        document=invalidation.DocumentLifecycleRef(
            concrete_model_label="apps_documents.rawtextdocument",
            document_pkid=43,
            document_id=uuid.uuid4(),
        ),
        source_hash="a" * 64,
        collection_id=17,
        figure_parent_identity=None,
    )

    with pytest.raises(RuntimeError, match="not part of the locked delete scope"):
        invalidation._assert_document_delete_scope(context, stale)


def test_document_graph_phase_helper_matches_task10_and_task11_order(monkeypatch):
    from apps.knowledge_graph.graph import invalidation

    document = _document_ref(invalidation).document
    row = object()
    actions = []
    collection_graph = invalidation.GraphRowLockSet(
        artifact_ids=(101,), run_ids=(201,)
    )
    document_graph = invalidation.GraphRowLockSet(
        artifact_ids=(103,), run_ids=(203,)
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_scopes",
        lambda ids, *, using: actions.append(("collection_advisory", ids)),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_rows",
        lambda ids, *, using: actions.append(("collection_rows", ids)) or (),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_collection_graph_rows",
        lambda ids, *, using: actions.append(("collection_graph", ids))
        or collection_graph,
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_document_scope",
        lambda document_id, *, using: actions.append(
            ("document_advisory", document_id)
        ),
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_document_graph_rows",
        lambda documents, *, using: actions.append(("document_graph", documents))
        or document_graph,
    )
    monkeypatch.setattr(
        invalidation,
        "_lock_exact_document_row",
        lambda ref, *, using: actions.append(("document_row", ref)) or row,
    )

    result = invalidation._lock_document_graph_phases(
        document,
        (17, 19),
        using="default",
    )

    assert actions == [
        ("collection_advisory", (17, 19)),
        ("collection_rows", (17, 19)),
        ("collection_graph", (17, 19)),
        ("document_advisory", document.document_id),
        ("document_graph", (document,)),
        ("document_row", document),
    ]
    assert result == (collection_graph, document_graph, row)
