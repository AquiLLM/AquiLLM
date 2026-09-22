from __future__ import annotations

import inspect

import pytest


def test_embedding_candidates_are_derived_only_from_exact_locked_endpoint_audit():
    from types import SimpleNamespace

    from apps.knowledge_graph.resolution import canonical

    def row(pk, collection_id, *, signature="local:model@locked", digest=None):
        return SimpleNamespace(
            pk=pk,
            artifact_id=collection_id * 100,
            collection_id=collection_id,
            entity_type="model",
            version_signature="",
            embedding_model_signature=signature,
            embedding_input_hash=digest or f"{pk:064x}",
            embedding=[1.0] * 1024,
        )

    candidates = canonical._derive_locked_embedding_candidates(
        (row(1, 10), row(2, 20)),
        artifact_embedding_signatures={
            1_000: "local:model@locked",
            2_000: "local:model@locked",
        },
    )
    mismatched_model = canonical._derive_locked_embedding_candidates(
        (row(1, 10), row(2, 20, signature="local:other@locked")),
        artifact_embedding_signatures={
            1_000: "local:model@locked",
            2_000: "local:other@locked",
        },
    )

    assert len(candidates) == 1
    assert candidates[0].embedding_model_signature == "local:model@locked"
    assert candidates[0].left_input_hash == f"{1:064x}"
    assert candidates[0].right_input_hash == f"{2:064x}"
    assert mismatched_model == ()
    with pytest.raises(RuntimeError, match="audit is incomplete"):
        canonical._derive_locked_embedding_candidates(
            (row(1, 10), row(2, 20, digest="poison")),
            artifact_embedding_signatures={
                1_000: "local:model@locked",
                2_000: "local:model@locked",
            },
        )
    assert set(inspect.signature(canonical.rebuild_canonical_registry).parameters) == {
        "using"
    }
