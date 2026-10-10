"""Receipts bind actual transport inputs to vectors without inventing identity."""

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

from aquillm import utils as facade
from lib.embeddings import local, multimodal


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def transport(monkeypatch, create):
    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda *_: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )


def test_indexed_receipts_bind_prepared_inputs_and_fitted_vectors(monkeypatch):
    monkeypatch.setenv("APP_EMBED_DIMS", "3")
    monkeypatch.setenv("APP_EMBED_MAX_INPUT_CHARS", "3")
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "declared-revision")
    transport(
        monkeypatch,
        lambda **_: SimpleNamespace(
            model="model-echo",
            data=[
                SimpleNamespace(index=1, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[1.0]),
            ],
        ),
    )
    results = facade.get_embedding_results(
        ["first", "second"], input_type="search_document"
    )
    assert [r.vector for r in results] == [[1, 0, 0], [2, 0, 0]]
    for result, prepared in zip(results, ["fir", "sec"]):
        receipt = result.provenance
        assert receipt["prepared_input_sha256"] == digest(prepared)
        assert receipt["role"] == "search_document"
        assert receipt["dimensions"] == {
            "raw": 1,
            "fitted": 3,
            "adaptation": "pad-zero",
        }
        assert receipt["input_transformations"] == ["character-truncation"]
        assert receipt["declared_identity"]["revision"] == "declared-revision"
        assert receipt["observed_identity"] == {
            "model": "model-echo",
            "revision": None,
            "precision": None,
        }
        assert (
            receipt["vector_sha256"]
            == hashlib.sha256(struct.pack("!3f", *result.vector)).hexdigest()
        )
        assert "first" not in json.dumps(receipt)


def test_context_retry_receipt_hashes_only_successful_sent_input(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        if len(calls) == 1:
            from .test_context_limit_handling import context_error
            raise context_error("maximum input length of 200 tokens")
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1, 2, 3])])

    transport(monkeypatch, create)
    monkeypatch.setenv("APP_EMBED_DIMS", "2")
    result = facade.get_embedding_result("a" * 300)
    assert len(calls) == 2
    assert result.provenance["prepared_input_sha256"] == digest("a" * 184)
    assert result.provenance["observed_identity"] == {
        "model": None,
        "revision": None,
        "precision": None,
    }
    assert result.provenance["dimensions"] == {
        "raw": 3,
        "fitted": 2,
        "adaptation": "truncate",
    }


def test_fallback_attributes_success_to_cohere(monkeypatch):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "legacy-cohere")
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "local-only-revision")
    transport(monkeypatch, lambda **_: (_ for _ in ()).throw(ConnectionError("offline")))
    client = SimpleNamespace(embed=lambda **_: SimpleNamespace(embeddings=[[1.0]]))
    monkeypatch.setattr(
        facade.apps, "get_app_config", lambda *_: SimpleNamespace(cohere_client=client)
    )
    result = facade.get_embedding_result("fallback")
    assert result.provenance["provider"] == "cohere"
    assert result.provenance["route"] == "cohere-embed"
    assert result.provenance["declared_identity"] == {
        "model": "embed-english-v3.0",
        "revision": None,
        "precision": None,
    }


def test_multimodal_receipt_binds_successful_route_without_raw_logging(monkeypatch):
    sent = []
    logs = []
    monkeypatch.setattr(
        multimodal,
        "logger",
        SimpleNamespace(
            debug=lambda *args, **kwargs: logs.append((args, kwargs)),
            info=lambda *args, **kwargs: logs.append((args, kwargs)),
        ),
    )

    def post(*_, **kwargs):
        sent.append(kwargs["json"])
        if len(sent) == 1:
            return SimpleNamespace(status_code=400, text="secret-provider-body")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"model": "echo", "data": [{"embedding": [1.0]}]},
        )

    monkeypatch.setattr(multimodal.requests, "post", post)
    result = facade.get_multimodal_embedding_result(
        "caption", "data:image/png;base64,private"
    )
    assert result.provenance["route"] == "vllm-openai-content"
    assert result.provenance["prepared_input_sha256"] == digest(sent[1]["input"])
    assert "private" not in json.dumps(result.provenance)
    assert "secret-provider-body" not in json.dumps(logs)
    assert "private" not in json.dumps(logs)


@pytest.mark.parametrize("route_attempt", [1, 2])
@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}],
        [{"index": 1, "embedding": [1.0]}],
        [{"index": -1, "embedding": [1.0]}],
        [{"index": True, "embedding": [1.0]}],
        [{"index": 0.0, "embedding": [1.0]}],
        [{"index": "0", "embedding": [1.0]}],
        [{"index": None, "embedding": [1.0]}],
        [],
        None,
        [None],
        [{"embedding": []}],
    ],
)
def test_multimodal_ambiguous_response_fails_closed_without_fallback(
    monkeypatch, route_attempt, rows
):
    from lib.embeddings.utils import EmbeddingContractError

    calls = []

    def post(*_, **kwargs):
        calls.append(kwargs["json"])
        assert len(calls) <= route_attempt, "ambiguous response retried another route"
        if len(calls) < route_attempt:
            return SimpleNamespace(status_code=400)
        return SimpleNamespace(status_code=200, json=lambda: {"data": rows})

    monkeypatch.setattr(multimodal.requests, "post", post)
    monkeypatch.setattr(
        facade,
        "get_embedding_result_via_local_openai",
        lambda *_, **__: pytest.fail("ambiguous multimodal response fell back to text"),
    )
    with pytest.raises(EmbeddingContractError):
        facade.get_multimodal_embedding_result(
            "caption", "data:image/png;base64,fixture"
        )
    assert len(calls) == route_attempt


@pytest.mark.parametrize("route_attempt", [1, 2])
@pytest.mark.parametrize("index", [{}, {"index": 0}])
def test_multimodal_singleton_binds_receipt_with_missing_or_zero_index(
    monkeypatch, route_attempt, index
):
    calls = []

    def post(*_, **kwargs):
        calls.append(kwargs["json"])
        if len(calls) < route_attempt:
            return SimpleNamespace(status_code=400)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"data": [{**index, "embedding": [1.0]}]},
        )

    monkeypatch.setattr(multimodal.requests, "post", post)
    monkeypatch.setenv("APP_EMBED_DIMS", "1")
    result = facade.get_multimodal_embedding_result(
        "caption", "data:image/png;base64,fixture"
    )
    assert result.vector == [1.0]
    assert len(calls) == route_attempt
    assert result.provenance["route"] == (
        "vllm-multi-modal-data" if route_attempt == 1 else "vllm-openai-content"
    )
    payload = calls[-1]
    prepared = (
        {"input": payload["input"], "multi_modal_data": payload["multi_modal_data"]}
        if route_attempt == 1
        else payload["input"]
    )
    assert result.provenance["prepared_input_sha256"] == digest(prepared)


def test_vector_integrity_survives_storage_roundtrip_and_rejects_replacement():
    from lib.embeddings.provenance import make_result, valid_provenance

    result = make_result(
        [0.1],
        provider="local-openai",
        route="openai-embeddings",
        role="search_query",
        prepared_input="x",
        original_input="x",
        model="model",
    )
    stored = list(struct.unpack("!f", struct.pack("!f", 0.1)))
    assert valid_provenance(stored, result.provenance) == result.provenance
    assert valid_provenance([0.2], result.provenance) is None
    assert valid_provenance(None, result.provenance) is None
    assert valid_provenance([0.1], {}) is None


def test_incomplete_receipt_cannot_be_counted_as_known_provenance():
    from lib.embeddings.provenance import valid_provenance

    partial = {
        "schema_version": 1,
        "dimensions": {"fitted": 1},
        "vector_sha256": hashlib.sha256(struct.pack("!f", 0.1)).hexdigest(),
    }
    assert valid_provenance([0.1], partial) is None


@pytest.mark.parametrize("modality", ["text", "image"])
def test_chunk_service_assigns_response_vector_and_receipt_together(
    monkeypatch, modality
):
    from apps.documents.services import chunk_embeddings
    from lib.embeddings.provenance import make_result

    response = make_result(
        [0.1],
        provider="local-openai",
        route="openai-embeddings",
        role="search_document",
        prepared_input="caption",
        original_input="caption",
        model="fixture",
    )
    monkeypatch.setattr(facade, "get_embedding", lambda *_, **__: response.vector)
    monkeypatch.setattr(
        facade, "get_multimodal_embedding", lambda *_, **__: response.vector
    )
    monkeypatch.setattr(facade, "get_embedding_result", lambda *_, **__: response)
    monkeypatch.setattr(
        facade, "get_multimodal_embedding_result", lambda *_, **__: response
    )
    monkeypatch.setattr(
        chunk_embeddings, "image_data_url", lambda _: "data:image/png;base64,fixture"
    )
    chunk = SimpleNamespace(
        modality=modality,
        Modality=SimpleNamespace(IMAGE="image"),
        content="caption",
        embedding=None,
        embedding_provenance=None,
    )
    chunk_embeddings.get_chunk_embedding(chunk)
    assert chunk.embedding == response.vector
    assert chunk.embedding_provenance == response.provenance
