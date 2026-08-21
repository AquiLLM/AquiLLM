from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from apps.knowledge_graph.projection import reconciler
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)


def test_pruning_is_bounded_and_dry_run_never_deletes(monkeypatch):
    rows = tuple(SimpleNamespace(id=uuid4()) for _ in range(3))
    monkeypatch.setattr(reconciler, "_prune_candidates", lambda **_kwargs: rows)
    monkeypatch.setattr(reconciler, "_orphan_generation_keys", lambda **_kwargs: ())
    monkeypatch.setattr(reconciler, "_projection_settings", lambda: object())
    monkeypatch.setattr(reconciler, "_postgres_repository", lambda: object())
    monkeypatch.setattr(reconciler, "_memgraph_repository", lambda: object())
    deleted = []
    monkeypatch.setattr(
        reconciler, "_delete_projection_generation", lambda row: deleted.append(row.id)
    )

    dry = reconciler.prune_graph_projection_generations(
        page_size=3, retain=1, dry_run=True
    )

    assert dry.candidate_count == 3 and deleted == []


def test_superseded_prune_loads_terminal_bundle_and_deletes_opaque_key() -> None:
    projection_id = uuid4()
    bundle = SimpleNamespace(generation=SimpleNamespace(generation_key="a" * 64))
    observed = []
    postgres = SimpleNamespace(
        load_projection_bundle=lambda **kwargs: (
            observed.append((kwargs["projection_id"], kwargs["purpose"])) or bundle
        )
    )
    graph = SimpleNamespace(
        delete_generation=lambda **kwargs: observed.append(
            (kwargs["generation_key"].value, kwargs["timeout_seconds"])
        )
    )

    reconciler._delete_projection_generation(
        row=SimpleNamespace(id=projection_id, state="superseded"),
        postgres=postgres,
        graph=graph,
        settings=SimpleNamespace(
            projection_batch_size=37,
            graph_overall_timeout_ms=500,
        ),
    )

    assert observed == [
        (projection_id, "prune"),
        ("a" * 64, 0.5),
    ]


def test_tombstoned_prune_derives_exact_hmac_generation_without_source(monkeypatch):
    generation = uuid4()
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    expected = codec.encode(
        ProjectionIdentifierDomain.COLLECTION,
        generation=generation,
        source=generation,
    ).value
    monkeypatch.setattr(reconciler, "projection_identifier_codec", lambda _value: codec)
    deleted = []

    reconciler._delete_projection_generation(
        row=SimpleNamespace(
            id=uuid4(),
            state="superseded",
            generation_key=generation,
            collection_id=None,
            artifact_id=None,
        ),
        postgres=SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("tombstone must not read deleted source")
            )
        ),
        graph=SimpleNamespace(
            delete_generation=lambda **kwargs: deleted.append(
                kwargs["generation_key"].value
            )
        ),
        settings=SimpleNamespace(graph_overall_timeout_ms=500),
    )

    assert deleted == [expected]
