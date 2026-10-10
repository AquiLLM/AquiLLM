"""Contract regressions at real provider/facade boundaries; transport is synthetic."""

from types import SimpleNamespace

import pytest

from aquillm import utils as facade
from lib.embeddings import local
from lib.embeddings.utils import fit_embedding_dims


@pytest.mark.parametrize(
    "vector",
    [[], [0.0] * 4, [float("nan")], [float("inf")], [True], ["1"], [0, 0, 0, 0, 1]],
)
def test_dimension_fit_rejects_invalid_or_zero_fitted_vector(monkeypatch, vector):
    monkeypatch.setenv("APP_EMBED_DIMS", "4")
    with pytest.raises(ValueError):
        fit_embedding_dims(vector)


def test_legacy_nonzero_dimension_adaptation_is_preserved(monkeypatch):
    monkeypatch.setenv("APP_EMBED_DIMS", "4")
    assert fit_embedding_dims([1, 2]) == [1, 2, 0, 0]
    assert fit_embedding_dims([1, 2, 3, 4, 5]) == [1, 2, 3, 4]


def provider(monkeypatch, rows):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=v) for i, v in rows]
        )

    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda *_: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    return calls


def test_batch_preserves_input_binding_when_response_order_changes(monkeypatch):
    provider(monkeypatch, [(1, [2.0]), (0, [1.0])])
    assert local.get_embeddings_via_local_openai(["first", "second"]) == [[1], [2]]


@pytest.mark.parametrize(
    "rows",
    [[(0, [1])], [(0, [1]), (0, [2])], [(0, [1]), (2, [2])], [(0, [1]), (True, [2])]],
)
def test_batch_rejects_missing_duplicate_or_invalid_indices(monkeypatch, rows):
    provider(monkeypatch, rows)
    with pytest.raises(ValueError):
        local.get_embeddings_via_local_openai(["first", "second"])


def test_local_role_is_accepted_without_changing_wire_format(monkeypatch):
    calls = provider(monkeypatch, [(0, [1.0])])
    monkeypatch.setenv("APP_EMBED_DIMS", "4")
    monkeypatch.setenv("APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE", "true")
    assert local.get_embedding_via_local_openai(
        "text", input_type="search_document"
    ) == [1]
    assert calls[0]["input"] == "text"
    assert set(calls[0]) == {"model", "input", "dimensions"}


@pytest.mark.parametrize("batch", [False, True])
def test_facade_propagates_role_internally(monkeypatch, batch):
    monkeypatch.setenv("APP_EMBED_DIMS", "4")

    def embed(value, *, input_type="search_query"):
        return [2.0] if input_type == "search_document" else [1.0]

    monkeypatch.setattr(facade, "get_embedding_via_local_openai", embed)
    monkeypatch.setattr(
        facade,
        "get_embeddings_via_local_openai",
        lambda values, **kwargs: [embed(v, **kwargs) for v in values],
    )
    result = (
        facade.get_embeddings(["text"], input_type="search_document")[0]
        if batch
        else facade.get_embedding("text", input_type="search_document")
    )
    assert result == [2, 0, 0, 0]


@pytest.mark.parametrize("batch", [False, True])
def test_contract_failure_cannot_cross_provider(monkeypatch, batch):
    provider(monkeypatch, [(0, [0.0])])
    monkeypatch.setattr(
        facade.apps,
        "get_app_config",
        lambda *_: pytest.fail("permanent invalid vector crossed into Cohere"),
    )
    with pytest.raises(ValueError):
        if batch:
            facade.get_embeddings(["text"])
        else:
            facade.get_embedding("text")


def test_transport_failure_retains_legacy_cohere_fallback_and_role(monkeypatch):
    monkeypatch.setenv("APP_EMBED_DIMS", "4")

    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic transport failure")

    def embed(*, texts, model, input_type):
        assert texts == ["text"]
        assert model == "embed-english-v3.0"
        return SimpleNamespace(
            embeddings=[[2 if input_type == "search_document" else 1]]
        )

    monkeypatch.setattr(facade, "get_embedding_via_local_openai", unavailable)
    monkeypatch.setattr(
        facade.apps,
        "get_app_config",
        lambda *_: SimpleNamespace(cohere_client=SimpleNamespace(embed=embed)),
    )
    assert facade.get_embedding("text", input_type="search_document") == [2, 0, 0, 0]


def test_chunk_contract_error_does_not_retry_or_assign_embedding(monkeypatch):
    from tenacity import stop_after_attempt, wait_none

    from apps.documents.services.chunk_embeddings import get_chunk_embedding

    provider(monkeypatch, [(0, [0.0])])
    chunk = SimpleNamespace(
        modality="text",
        Modality=SimpleNamespace(IMAGE="image"),
        content="text",
        embedding=None,
    )
    attempts = []
    # Bound the baseline's otherwise infinite retry so the regression fails promptly.
    bounded = get_chunk_embedding.retry_with(
        stop=stop_after_attempt(2),
        wait=wait_none(),
        before=lambda _: attempts.append(1),
    )
    with pytest.raises(ValueError):
        bounded(chunk)
    assert attempts == [1]
    assert chunk.embedding is None


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("vectors", [[], [[0.0]], [[1.0], [2.0]]])
def test_cohere_invalid_response_remains_a_permanent_error(monkeypatch, batch, vectors):
    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic transport failure")

    monkeypatch.setattr(facade, "get_embedding_via_local_openai", unavailable)
    monkeypatch.setattr(facade, "get_embeddings_via_local_openai", unavailable)
    monkeypatch.setattr(
        facade.apps,
        "get_app_config",
        lambda *_: SimpleNamespace(
            cohere_client=SimpleNamespace(
                embed=lambda **_: SimpleNamespace(embeddings=vectors)
            )
        ),
    )
    with pytest.raises(ValueError):
        if batch:
            facade.get_embeddings(["text"])
        else:
            facade.get_embedding("text")


def test_strict_index_cannot_persist_zero_vector(monkeypatch):
    monkeypatch.setattr(facade, "strict_index_embedding_signature", lambda: "synthetic")
    monkeypatch.setattr(
        facade,
        "get_strict_indexed_embeddings_via_local_openai",
        lambda _: [(0, [0.0] * 1024)],
    )
    with pytest.raises(ValueError):
        facade.get_strict_index_embeddings(
            ["text"], expected_model_signature="synthetic"
        )
