"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_projection_reconciliation_control import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
    SimpleNamespace,
    _bundle,
    _manifest,
    _settings,
    generation_audit,
    nullcontext,
    reconciler,
    uuid4,
)


def test_reconcile_replays_only_missing_expired_or_drifted_work(monkeypatch):
    rows = {
        11: SimpleNamespace(id=uuid4(), state="ready"),
        22: SimpleNamespace(id=uuid4(), state="building"),
        33: SimpleNamespace(id=uuid4(), state="ready"),
    }
    reasons = {11: None, 22: "expired_lease", 33: "checksum_drift"}
    pages = [((1, 11), (2, 22), (3, 33)), ()]
    monkeypatch.setattr(
        reconciler, "_active_artifact_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(
        reconciler,
        "_projection_for_active",
        lambda **kwargs: rows[kwargs["artifact_id"]],
    )
    monkeypatch.setattr(
        reconciler,
        "_generation_audit",
        lambda **kwargs: SimpleNamespace(
            replay_reason=reasons[
                next(key for key, value in rows.items() if value is kwargs["row"])
            ]
        ),
    )
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(reconciler, "_projection_settings", _settings)
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda _value: object()
    )
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_atomic", lambda _using: nullcontext())
    enqueued = []
    superseded = []
    monkeypatch.setattr(
        reconciler,
        "supersede_projection_locked",
        lambda **kwargs: superseded.append(kwargs["projection_id"]),
    )
    monkeypatch.setattr(
        reconciler,
        "enqueue_collection_projection_locked",
        lambda **kwargs: enqueued.append(kwargs["artifact_id"]),
    )

    summary = reconciler.reconcile_graph_projections(page_size=3, dry_run=False)

    assert enqueued == [22, 33]
    assert superseded == [rows[33].id]
    assert summary.enqueued_count == 2
    assert summary.drift_count == 1
    assert summary.replayed_count == 2



def test_global_orphan_scan_uses_exclusive_opaque_cursor(monkeypatch):
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v7")
    generation = uuid4()
    authoritative = _bundle(
        codec.encode(
            ProjectionIdentifierDomain.COLLECTION,
            generation=generation,
            source=generation,
        ).value
    )
    row = SimpleNamespace(
        id=1, state="ready", generation_key=generation, identifier_key_version="key-v7"
    )
    monkeypatch.setattr(
        generation_audit, "projection_identifier_codec", lambda *args, **kwargs: codec
    )
    projection_pages = [(row,), ()]
    monkeypatch.setattr(
        generation_audit,
        "_projection_page",
        lambda **_kwargs: projection_pages.pop(0),
    )
    postgres = SimpleNamespace(load_projection_bundle=lambda **_kwargs: authoritative)
    observed_cursors = []

    def list_generations(**kwargs):
        observed_cursors.append(kwargs["after_generation_key"])
        if kwargs["after_generation_key"] is None:
            return (_manifest(authoritative), _manifest(_bundle("2" * 64)))
        return ()

    graph = SimpleNamespace(list_generations=list_generations)
    settings = _settings()
    settings.projection_batch_size = 2

    orphaned = generation_audit.orphan_generation_keys(
        postgres=postgres,
        graph=graph,
        settings=settings,
        limit=10,
    )

    assert tuple(key.value for key in orphaned) == ("2" * 64,)
    assert observed_cursors[0] is None
    assert observed_cursors[1].value == "2" * 64



def test_collection_orphan_scan_never_reads_other_collection_manifests(monkeypatch):
    authoritative = _bundle("1" * 64)
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v7")
    projection_pages = [
        (
            SimpleNamespace(
                id=1,
                state="ready",
                generation_key=uuid4(),
                identifier_key_version="key-v7",
            ),
        ),
        (),
    ]
    monkeypatch.setattr(
        generation_audit, "projection_identifier_codec", lambda *args, **kwargs: codec
    )
    monkeypatch.setattr(
        generation_audit,
        "_projection_page",
        lambda **kwargs: (
            projection_pages.pop(0) if kwargs["collection_id"] == 7 else ()
        ),
    )
    observed = []
    graph = SimpleNamespace(
        list_generations=lambda **kwargs: observed.append(kwargs) or ()
    )

    generation_audit.orphan_generation_keys(
        postgres=SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: authoritative
        ),
        graph=graph,
        settings=_settings(),
        limit=10,
        collection_id=7,
        collection_key=generation_audit._opaque_generation("e" * 64),
    )

    assert observed[0]["collection_key"].value == "e" * 64
