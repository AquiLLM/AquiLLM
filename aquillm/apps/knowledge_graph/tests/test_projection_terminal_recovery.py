from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import generation_audit, reconciler, runtime
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.tests.test_projection_runtime import _projection_environment


def _settings():
    return SimpleNamespace(
        projection_batch_size=25,
        graph_overall_timeout_ms=500,
    )


@pytest.mark.parametrize("source_change", ("newer_artifact", "newer_membership"))
def test_terminal_prune_uses_immutable_orm_authority_after_source_change(
    monkeypatch, source_change
):
    generation = uuid4()
    row = SimpleNamespace(
        id=uuid4(),
        state="superseded",
        generation_key=generation,
        collection_id=7,
        artifact_id=11,
        membership_epoch=3,
        membership_checksum="a" * 64,
        identifier_key_version="key-v1",
    )
    filters = []

    class Query:
        def using(self, _alias):
            return self

        def filter(self, **kwargs):
            filters.append(kwargs)
            return self

        def order_by(self, *_args):
            return self

        def __getitem__(self, _value):
            return (row,)

    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    expected = codec.encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=generation,
    ).value
    deleted = []
    monkeypatch.setattr(reconciler.CollectionGraphProjection, "objects", Query())
    monkeypatch.setattr(reconciler, "_projection_settings", _settings)
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda _s, **_kwargs: codec
    )
    monkeypatch.setattr(
        reconciler,
        "_postgres_repository",
        lambda: (_ for _ in ()).throw(
            AssertionError(f"prune opened mutable {source_change} source")
        ),
    )
    monkeypatch.setattr(
        reconciler,
        "_memgraph_repository",
        lambda: SimpleNamespace(
            delete_generation=lambda **kwargs: deleted.append(
                kwargs["generation_key"].value
            )
        ),
    )
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())

    summary = reconciler.prune_graph_projection_generations(
        projection_id=row.id,
        page_size=10,
        retain=2,
        dry_run=False,
    )

    assert filters == [
        {"state__in": ("failed", "superseded")},
        {"pk": row.id},
    ]
    assert deleted == [expected]
    assert summary.deleted_count == 1


def test_terminal_generation_identity_never_reloads_mutable_bundle(monkeypatch):
    generation = uuid4()
    rows = (
        SimpleNamespace(id=1, state="pending"),
        SimpleNamespace(id=2, state="building"),
        SimpleNamespace(id=3, state="ready"),
        SimpleNamespace(
            id=4,
            state="superseded",
            generation_key=generation,
            collection_id=7,
            artifact_id=11,
            identifier_key_version="key-v1",
        ),
    )
    pages = [rows, ()]
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    expected = codec.encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=generation,
    ).value
    monkeypatch.setattr(
        generation_audit, "_projection_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(
        generation_audit,
        "projection_identifier_codec",
        lambda _settings, **_kwargs: codec,
        raising=False,
    )

    observed = []

    def load(**kwargs):
        observed.append((kwargs["projection_id"], kwargs["purpose"]))
        return SimpleNamespace(
            generation=SimpleNamespace(generation_key=str(kwargs["projection_id"]) * 64)
        )

    keys = generation_audit._authoritative_generation_keys(
        postgres=SimpleNamespace(load_projection_bundle=load),
        settings=_settings(),
        collection_id=None,
    )

    assert observed == [(2, "build"), (3, "audit")]
    assert keys == frozenset(("2" * 64, "3" * 64, expected))


def test_direct_prune_refuses_nonterminal_authority(monkeypatch):
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    monkeypatch.setattr(reconciler, "projection_identifier_codec", lambda _s: codec)

    with pytest.raises(ValueError, match="terminal"):
        reconciler._delete_projection_generation(
            row=SimpleNamespace(state="ready", generation_key=uuid4()),
            graph=SimpleNamespace(
                delete_generation=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("active generation must not be deleted")
                )
            ),
            settings=_settings(),
        )


def test_rotated_runtime_prunes_with_persisted_generation_key_version():
    generation = uuid4()
    environment = {
        **_projection_environment(),
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v2",
    }
    settings = runtime.load_projection_runtime_settings(environment)
    expected = HmacSha256ProjectionIdentifierCodec(
        b"identifier-secret", key_version="key-v1"
    ).encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=generation,
    )
    deleted = []

    reconciler._delete_projection_generation(
        row=SimpleNamespace(
            state="superseded",
            generation_key=generation,
            identifier_key_version="key-v1",
        ),
        graph=SimpleNamespace(
            delete_generation=lambda **kwargs: deleted.append(kwargs["generation_key"])
        ),
        settings=settings,
    )

    assert deleted == [expected]


def test_rotated_runtime_orphan_scan_uses_persisted_generation_key_version(
    monkeypatch,
):
    generation = uuid4()
    row = SimpleNamespace(
        id=uuid4(),
        state="superseded",
        generation_key=generation,
        identifier_key_version="key-v1",
    )
    pages = [(row,), ()]
    environment = {
        **_projection_environment(),
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "key-v2",
    }
    settings = runtime.load_projection_runtime_settings(environment)
    expected = HmacSha256ProjectionIdentifierCodec(
        b"identifier-secret", key_version="key-v1"
    ).encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=generation,
    )
    monkeypatch.setattr(
        generation_audit, "_projection_page", lambda **_kwargs: pages.pop(0)
    )

    keys = generation_audit._authoritative_generation_keys(
        postgres=SimpleNamespace(),
        settings=settings,
        collection_id=None,
    )

    assert keys == frozenset((expected.value,))


def test_reconcile_replaces_cas_supersession_and_preserves_terminal_prune(
    monkeypatch,
):
    terminal = SimpleNamespace(id=uuid4(), state="superseded")
    filters = []

    class Query:
        def using(self, _alias):
            return self

        def filter(self, **kwargs):
            filters.append(kwargs)
            return self

        def order_by(self, *_args):
            return self

        def first(self):
            active_states = ("pending", "building", "ready")
            return (
                None
                if any(row.get("state__in") == active_states for row in filters)
                else terminal
            )

    pages = [((7, 11),), ()]
    monkeypatch.setattr(reconciler.CollectionGraphProjection, "objects", Query())
    monkeypatch.setattr(
        reconciler, "_active_artifact_page", lambda **_kwargs: pages.pop(0)
    )
    monkeypatch.setattr(reconciler, "_projection_settings", _settings)
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(
        reconciler, "projection_identifier_codec", lambda _settings: object()
    )
    monkeypatch.setattr(reconciler, "_atomic", lambda _using: nullcontext())
    enqueued = []
    resuperseded = []
    monkeypatch.setattr(
        reconciler,
        "supersede_projection_locked",
        lambda **kwargs: resuperseded.append(kwargs["projection_id"]),
    )
    monkeypatch.setattr(
        reconciler,
        "enqueue_collection_projection_locked",
        lambda **kwargs: enqueued.append(
            (kwargs["collection_id"], kwargs["artifact_id"])
        ),
    )

    summary = reconciler.reconcile_graph_projections(page_size=1, dry_run=False)

    assert filters == [
        {
            "collection_pk_snapshot": 7,
            "artifact_pk_snapshot": 11,
            "state__in": ("pending", "building", "ready"),
        }
    ]
    assert enqueued == [(7, 11)]
    assert resuperseded == []
    assert summary.replayed_count == summary.enqueued_count == 1
