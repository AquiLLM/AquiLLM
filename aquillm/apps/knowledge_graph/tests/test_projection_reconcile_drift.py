"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_projection_reconciler import (
    SimpleNamespace,
    nullcontext,
    pytest,
    reconciler,
)


@pytest.mark.parametrize("unexpected_failure", [False, True])
def test_reconcile_handles_empty_store_drift_and_newer_artifact_in_pages(
    monkeypatch, unexpected_failure
):
    pages = [((1, 11), (2, 22)), ((3, 33),), ()]
    monkeypatch.setattr(
        reconciler, "_active_artifact_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(reconciler, "_atomic", lambda _using: nullcontext())
    monkeypatch.setattr(reconciler, "_projection_for_active", lambda **_kwargs: None)
    monkeypatch.setattr(
        reconciler,
        "_generation_audit",
        lambda **_kwargs: SimpleNamespace(replay_reason="missing_authority"),
    )
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: object())
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda _value: object()
    )
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    enqueued = []

    def enqueue(**kwargs):
        if unexpected_failure:
            raise RuntimeError("state backend failed")
        enqueued.append((kwargs["collection_id"], kwargs["artifact_id"]))

    monkeypatch.setattr(reconciler, "enqueue_collection_projection_locked", enqueue)
    if unexpected_failure:
        with pytest.raises(RuntimeError, match="state backend failed"):
            reconciler.reconcile_graph_projections(page_size=2, dry_run=False)
        return
    summary = reconciler.reconcile_graph_projections(page_size=2, dry_run=False)

    assert summary.examined_count == 3
    assert enqueued == [(1, 11), (2, 22), (3, 33)]
