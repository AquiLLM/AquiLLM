from __future__ import annotations

import inspect

import pytest

from apps.knowledge_graph.retrieval import query_embedding

SIGNATURE = (
    "local-openai:model@revision:endpoint=" + "a" * 64 + ":dims=1024:prep=kg-entity-v1"
)


def test_embedding_requires_exact_signature_dimension_and_finite_values(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def api():
        def signature():
            return SIGNATURE

        def embed(queries, *, expected_model_signature):
            calls.append((queries[0], expected_model_signature))
            return ([(0, [0.0] * 1024)], expected_model_signature)

        return signature, embed

    monkeypatch.setattr(query_embedding, "_load_embedding_api", api)
    monkeypatch.setattr(query_embedding, "monotonic", lambda: 1.0)

    vector = query_embedding.embed_unresolved_query_span(
        text="transient model", expected_signature=SIGNATURE, deadline=2.0
    )

    assert len(vector) == 1024
    assert calls == [("transient model", SIGNATURE)]


@pytest.mark.parametrize(
    ("actual_signature", "vector"),
    (
        ("other", [0.0] * 1024),
        (SIGNATURE, [0.0] * 1023),
        (SIGNATURE, [float("nan")] * 1024),
    ),
)
def test_embedding_rejects_signature_dimension_or_numeric_drift(
    monkeypatch, actual_signature, vector
) -> None:
    monkeypatch.setattr(
        query_embedding,
        "_load_embedding_api",
        lambda: (
            lambda: actual_signature,
            lambda _queries, **_kwargs: ([(0, vector)], actual_signature),
        ),
    )
    monkeypatch.setattr(query_embedding, "monotonic", lambda: 1.0)
    with pytest.raises(RuntimeError):
        query_embedding.embed_unresolved_query_span(
            text="model", expected_signature=SIGNATURE, deadline=2.0
        )


def test_expired_deadline_never_imports_or_calls_embedding(monkeypatch) -> None:
    monkeypatch.setattr(query_embedding, "monotonic", lambda: 2.0)
    monkeypatch.setattr(
        query_embedding,
        "_load_embedding_api",
        lambda: (_ for _ in ()).throw(AssertionError("embedding reached")),
    )
    with pytest.raises(TimeoutError):
        query_embedding.embed_unresolved_query_span(
            text="model", expected_signature=SIGNATURE, deadline=2.0
        )


def test_query_embedding_has_no_persistence_operations() -> None:
    source = inspect.getsource(query_embedding)
    for forbidden in (".save(", ".create(", ".update(", "bulk_create"):
        assert forbidden not in source
