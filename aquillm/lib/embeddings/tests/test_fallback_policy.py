"""Ordinary embedding policy at real provider boundaries; no network requests."""

import traceback
from types import SimpleNamespace

import pytest

from aquillm import utils as facade
from lib.embeddings import audit, local
from lib.embeddings.utils import EmbeddingContractError


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.delenv("APP_EMBED_FALLBACK_POLICY", raising=False)
    monkeypatch.setenv("APP_EMBED_DIMS", "4")
    monkeypatch.setenv("APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE", "1")
    calls = {"local": [], "cohere": []}

    def create(**kwargs):
        calls["local"].append(kwargs)
        if calls.get("fail"):
            raise ConnectionError("PRIVATE_URL PRIVATE_QUERY PRIVATE_KEY")
        count = len(kwargs["input"]) if isinstance(kwargs["input"], list) else 1
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[1.0]) for i in range(count)]
        )

    def embed(**kwargs):
        calls["cohere"].append(kwargs)
        return SimpleNamespace(embeddings=[[2.0] for _ in kwargs["texts"]])

    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda *_: SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        facade.apps,
        "get_app_config",
        lambda *_: SimpleNamespace(cohere_client=SimpleNamespace(embed=embed)),
    )
    monkeypatch.setattr(
        facade, "get_multimodal_embedding_via_vllm_pooling", lambda *_: None
    )
    return calls


def invoke(path):
    if path == "batch":
        return facade.get_embeddings(["synthetic"], input_type="search_document")[0]
    if path == "multimodal":
        return facade.get_multimodal_embedding(
            "synthetic", "data:image/png;base64,AAAA"
        )
    return facade.get_embedding("synthetic", input_type="search_document")


@pytest.mark.parametrize("path", ["single", "batch", "multimodal"])
@pytest.mark.parametrize("policy", [None, "legacy-cohere"])
def test_legacy_policy_returns_sentinel_cohere_vector(
    monkeypatch, transport, path, policy
):
    if policy is not None:
        monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", policy)
    transport["fail"] = True
    assert invoke(path) == [2, 0, 0, 0]
    assert transport["cohere"][0]["input_type"] == "search_document"


@pytest.mark.parametrize("path", ["single", "batch", "multimodal"])
def test_local_only_outage_never_calls_cohere_and_is_redacted(
    monkeypatch, transport, path
):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")
    transport["fail"] = True
    with pytest.raises(RuntimeError) as caught:
        invoke(path)
    assert type(caught.value).__name__ == "EmbeddingUpstreamUnavailableError"
    assert not isinstance(caught.value, EmbeddingContractError)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    assert transport["cohere"] == []


@pytest.mark.parametrize("path", ["single", "batch", "multimodal"])
def test_local_only_success_preserves_local_input_and_dimensions(
    monkeypatch, transport, path
):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")
    assert invoke(path) == [1, 0, 0, 0]
    assert transport["local"][0]["input"] == (
        ["synthetic"] if path == "batch" else "synthetic"
    )
    assert transport["local"][0]["dimensions"] == 4
    assert set(transport["local"][0]) == {"model", "input", "dimensions"}
    assert transport["cohere"] == []


@pytest.mark.parametrize(
    "policy", ["", " ", "LOCAL-ONLY", "local-only ", "PRIVATE_TYPO"]
)
@pytest.mark.parametrize("path", ["single", "batch", "multimodal"])
def test_invalid_policy_fails_before_any_provider(monkeypatch, transport, policy, path):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", policy)
    monkeypatch.setattr(
        facade,
        "get_multimodal_embedding_via_vllm_pooling",
        lambda *_: pytest.fail("invalid configuration reached pooling"),
    )
    with pytest.raises(EmbeddingContractError) as caught:
        invoke(path)
    assert "PRIVATE" not in str(caught.value)
    assert transport["local"] == transport["cohere"] == []


def test_invalid_policy_rejects_empty_batch(monkeypatch):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "")
    with pytest.raises(EmbeddingContractError):
        facade.get_embeddings([])


@pytest.mark.parametrize("policy", ["local-only", "legacy-cohere"])
def test_audit_reports_selected_policy_without_compatibility_claim(monkeypatch, policy):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", policy)
    report = audit.build_report()
    assert report["declared"]["transport_failure_policy"] == policy
    assert report["compatibility"] == "unproven"
    assert report["historical_identity"] == "unknown"


def test_audit_invalid_policy_fails_before_probe(monkeypatch):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "")
    monkeypatch.setattr(
        audit, "OpenAI", lambda **_: pytest.fail("invalid policy probed")
    )
    with pytest.raises(EmbeddingContractError):
        audit.build_report(probe=True)


def test_invalid_policy_does_not_retry_or_assign_chunk(monkeypatch, transport):
    from apps.documents.services.chunk_embeddings import get_chunk_embedding

    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "")
    chunk = SimpleNamespace(
        modality="text",
        Modality=SimpleNamespace(IMAGE="image"),
        content="synthetic",
        embedding=None,
    )
    with pytest.raises(EmbeddingContractError):
        get_chunk_embedding(chunk)
    assert chunk.embedding is None
    assert transport["local"] == transport["cohere"] == []


def test_multimodal_transport_error_does_not_amplify_to_text(monkeypatch, transport):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")

    def unavailable(*_):
        raise ConnectionError("PRIVATE_POOLING_FAILURE")

    monkeypatch.setattr(
        facade, "get_multimodal_embedding_via_vllm_pooling", unavailable
    )
    with pytest.raises(facade.EmbeddingUpstreamUnavailableError):
        invoke("multimodal")
    assert transport["local"] == []
    assert transport["cohere"] == []


@pytest.mark.parametrize("path", ["single", "batch", "multimodal"])
def test_local_only_contract_failure_is_still_permanent(monkeypatch, transport, path):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")
    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda *_: SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.0])]
                )
            )
        ),
    )
    with pytest.raises(EmbeddingContractError):
        invoke(path)
    assert transport["cohere"] == []
