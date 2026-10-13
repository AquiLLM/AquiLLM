from hashlib import sha256
from types import SimpleNamespace

import pytest

from lib.embeddings import document_provider as provider
from lib.embeddings.document_contracts import build_budget
from lib.embeddings.provenance import input_digest, valid_provenance
from lib.embeddings.tests.test_document_contracts import profile
from lib.embeddings.utils import EmbeddingContractError
from lib.parsers.source_contracts import SourceChunkV1
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def chunk(text="source"):
    return SourceChunkV1(
        0,
        len(text),
        text,
        text,
        (),
        sha256(text.encode()).hexdigest(),
        input_digest(text),
    )


@pytest.fixture
def transport(monkeypatch):
    calls = []
    monkeypatch.setenv("APP_EMBED_BASE_URL", "http://fake-models:8000/v1")

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="synthetic",
            data=[
                SimpleNamespace(index=i, embedding=[float(i + 1)] * 1024)
                for i in reversed(range(len(kwargs["input"])))
            ],
        )

    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(provider, "_endpoint", lambda: "http://fake-models:8000/v1")
    space = profile()
    provider.register_verified_profile(
        space,
        endpoint="http://fake-models:8000/v1",
        deployment_artifact_sha256="c" * 64,
        engine_manifest=provider.engine_manifest_for(space),
        query_template="{text}",
        document_template="{text}",
    )
    return space, calls


def test_shuffled_results_retain_receipts_and_order(transport):
    space, calls = transport
    results = provider.embed_document_batch(
        (chunk("one"), chunk("two")), space=space, budget=build_budget(120000)
    )
    assert [r.vector[0] for r in results] == [1, 2]
    assert results[0].provenance["observed_identity"]["revision"] is None
    assert valid_provenance(results[0].vector, results[0].provenance)
    assert results[0].provenance["prepared_input_sha256"] == input_digest("one")
    assert len(calls) == 1


@pytest.mark.parametrize("indices", [(0, 0), (False, 1), (0,), (0, 2)])
def test_invalid_response_indices_do_not_fallback(transport, monkeypatch, indices):
    space, calls = transport
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                model="synthetic",
                data=[
                    SimpleNamespace(index=i, embedding=[1.0] * 1024) for i in indices
                ],
            )
        )
    )
    monkeypatch.setattr(provider, "_client", lambda: client)
    with pytest.raises(EmbeddingContractError):
        provider.embed_document_batch(
            (chunk("one"), chunk("two")), space=space, budget=build_budget(120000)
        )


@pytest.mark.parametrize("model", ["foreign", None])
def test_unknown_response_never_creates_compatible_observation(
    transport, monkeypatch, model
):
    space, _ = transport
    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    model=model, data=[SimpleNamespace(index=0, embedding=[1.0] * 1024)]
                )
            )
        ),
    )
    if model is None:
        result = provider.observe_query_embedding(
            "query", space=space, budget=TurnBudget(TurnLimits())
        )
        assert result.compatibility_key is None
        assert result.receipt["observed_identity"]["model"] is None
    else:
        with pytest.raises(EmbeddingContractError):
            provider.observe_query_embedding(
                "query", space=space, budget=TurnBudget(TurnLimits())
            )


def test_build_can_dispatch_after_15_seconds_query_cannot(transport):
    space, calls = transport
    now = [0.0]
    build = build_budget(120000, clock=lambda: now[0])
    query = TurnBudget(TurnLimits(), clock=lambda: now[0])
    now[0] = 16
    provider.embed_document_batch((chunk(),), space=space, budget=build)
    assert calls[-1]["timeout"] == 30
    with pytest.raises(TimeoutError):
        provider.observe_query_embedding("query", space=space, budget=query)
    now[0] = 120
    with pytest.raises(TimeoutError):
        provider.embed_document_batch((chunk(),), space=space, budget=build)
    assert len(calls) == 1


def test_endpoint_binding_is_separate_and_required(transport, monkeypatch):
    space, calls = transport
    monkeypatch.setattr(provider, "_endpoint", lambda: "http://foreign/v1")
    with pytest.raises(EmbeddingContractError, match="endpoint_binding_mismatch"):
        provider.embed_document_batch(
            (chunk(),), space=space, budget=build_budget(120000)
        )
    assert not calls


def test_no_prefix_shrink_or_cross_provider_on_context_error(transport, monkeypatch):
    import httpx
    from openai import BadRequestError

    space, calls = transport
    dispatched = []

    def reject(**kwargs):
        dispatched.append(kwargs["input"])
        raise BadRequestError(
            "context limit",
            response=httpx.Response(
                400, request=httpx.Request("POST", "http://fake/v1")
            ),
            body={},
        )

    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(embeddings=SimpleNamespace(create=reject)),
    )
    with pytest.raises(EmbeddingContractError):
        provider.embed_document_batch(
            (chunk("unchanged input"),), space=space, budget=build_budget(120000)
        )
    assert dispatched == [["unchanged input"]]


def test_wrong_dimensions_and_provider_receipt_rejected(transport):
    from lib.embeddings.provenance import make_result

    space, _ = transport
    for provider_name, dims in [("cohere", 1024), ("local-openai", 512)]:
        result = make_result(
            [1.0] * dims,
            provider=provider_name,
            route="cohere-embed" if provider_name == "cohere" else "openai-embeddings",
            role="search_document",
            prepared_input="source",
            original_input="source",
            model=space.model,
            response_model=space.model,
        )
        with pytest.raises(EmbeddingContractError):
            provider.compatible_receipt(
                result.vector,
                result.provenance,
                space=space,
                role="search_document",
                prepared_input_sha256=input_digest("source"),
            )


def test_media_bytes_are_digest_checked_and_wire_bound(transport, monkeypatch):
    from dataclasses import replace
    from io import BytesIO
    from PIL import Image
    from django.core.files.storage import default_storage
    from lib.parsers.source_contracts import source_artifact_ref

    space, _ = transport

    def pixels(color):
        result = BytesIO()
        Image.new("RGB", (2, 2), color).save(result, "PNG")
        return result.getvalue()

    red, blue = pixels("red"), pixels("blue")
    refs = [
        source_artifact_ref(data, member_path=None, container_sha256=None)
        for data in (red, blue)
    ]
    blobs = dict(zip((ref.storage_key for ref in refs), (red, blue)))
    monkeypatch.setattr(default_storage, "open", lambda key, mode: BytesIO(blobs[key]))
    first = replace(chunk(), modality="image", media_artifact=refs[0])
    second = replace(chunk(), modality="image", media_artifact=refs[1])
    assert (
        provider.prepare_document(first, space=space)[1]
        != provider.prepare_document(second, space=space)[1]
    )
    blobs[refs[0].storage_key] = b"corrupt"
    with pytest.raises(EmbeddingContractError, match="media_unavailable"):
        provider.embed_document_batch(
            (first,), space=space, budget=build_budget(120000)
        )
    caption = replace(first, modality="caption_only")
    assert provider.prepare_document(caption, space=space)[0] == "source"


def test_client_setup_cannot_dispatch_after_original_deadline(transport, monkeypatch):
    space, calls = transport
    now = [0.0]
    budget = build_budget(1000, clock=lambda: now[0])
    original = provider._client

    def late_client():
        now[0] = 2.0
        return original()

    monkeypatch.setattr(provider, "_client", late_client)
    with pytest.raises(TimeoutError):
        provider.embed_document_batch((chunk(),), space=space, budget=budget)
    assert not calls


def test_query_cannot_borrow_indexing_deadline(transport):
    space, calls = transport
    with pytest.raises(TypeError):
        provider.observe_query_embedding(
            "query", space=space, budget=build_budget(120000)
        )
    assert not calls
