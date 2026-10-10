"""Finite provider work and precise failure classes; no external network."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import httpx

from aquillm import utils as facade
from lib.embeddings import local, cohere, multimodal
from lib.embeddings.utils import EmbeddingContractError


def test_local_client_disables_sdk_retry_and_bounds_timeout(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(local, "OpenAI", factory)
    monkeypatch.setattr(local, "_LOCAL_OPENAI_CLIENT", None)
    local._get_local_openai_client("http://localhost/v1", "synthetic")
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["timeout"] == 30


def test_context_retry_limit_cannot_be_unbounded(monkeypatch):
    monkeypatch.setenv("APP_EMBED_CONTEXT_RETRIES", "1000")
    from .test_context_limit_handling import context_error
    create = Mock(side_effect=context_error("maximum input length of 10000000 tokens"))
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(EmbeddingContractError):
        local._embed_local_with_context_retry(client, "synthetic", "x" * 1000000)
    assert create.call_count == 6


def test_cohere_bounds_sdk_work():
    embed = Mock(return_value=SimpleNamespace(embeddings=[[1.0]]))
    cohere.get_embedding_results_via_cohere(SimpleNamespace(embed=embed), ["synthetic"], "search_document")
    assert embed.call_args.kwargs["request_options"] == {"max_retries": 0, "timeout_in_seconds": 30}
    assert embed.call_args.kwargs["batching"] is False


def test_cohere_large_batch_is_sequential_and_stops_on_outage():
    sizes = []
    def embed(**kwargs):
        sizes.append(len(kwargs["texts"]))
        if len(sizes) == 2:
            raise ConnectionError("offline")
        return SimpleNamespace(embeddings=[[1.0] for _ in kwargs["texts"]])
    with pytest.raises(ConnectionError):
        cohere.get_embedding_results_via_cohere(SimpleNamespace(embed=embed), ["synthetic"] * 250, "search_document")
    assert sizes == [96, 96]


@pytest.mark.parametrize("failure", [TypeError("bug"), ValueError("bug")])
@pytest.mark.parametrize("batch", [False, True])
def test_programming_failure_never_becomes_transient_or_falls_back(monkeypatch, failure, batch):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "legacy-cohere")
    route = "get_embedding_results_via_local_openai" if batch else "get_embedding_result_via_local_openai"
    monkeypatch.setattr(facade, route, Mock(side_effect=failure))
    fallback = Mock()
    monkeypatch.setattr(facade, "get_embedding_results_via_cohere", fallback)
    monkeypatch.setattr(facade, "get_embedding_result_via_cohere", fallback)
    with pytest.raises(type(failure), match="bug"):
        (facade.get_embedding_results(["synthetic"]) if batch else facade.get_embedding_result("synthetic"))
    fallback.assert_not_called()


@pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), None])
def test_multimodal_outage_stops_after_one_request(monkeypatch, failure):
    post = Mock(side_effect=failure, return_value=SimpleNamespace(status_code=503))
    monkeypatch.setattr(multimodal.requests, "post", post)
    monkeypatch.setattr(facade, "get_embedding_result_via_local_openai", Mock(side_effect=facade.EmbeddingUpstreamUnavailableError("offline")))
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")
    with pytest.raises(facade.EmbeddingUpstreamUnavailableError):
        facade.get_multimodal_embedding_result("synthetic", "data:image/png;base64,AAAA")
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 30


def test_chunk_invocation_does_not_retry_transient_or_programming_error(monkeypatch):
    from apps.documents.services.chunk_embeddings import get_chunk_embedding
    from tenacity import stop_after_attempt, wait_none

    # Bound the OLD decorator while reproducing its amplification, so RED cannot hang.
    invoke = get_chunk_embedding.retry_with(stop=stop_after_attempt(2), wait=wait_none()) if hasattr(get_chunk_embedding, "retry_with") else get_chunk_embedding
    for failure in [facade.EmbeddingUpstreamUnavailableError("offline"), TypeError("bug")]:
        provider = Mock(side_effect=failure)
        monkeypatch.setattr(facade, "get_embedding_result", provider)
        chunk = SimpleNamespace(modality="text", Modality=SimpleNamespace(IMAGE="image"), content="synthetic", embedding=None)
        with pytest.raises(type(failure)):
            invoke(chunk)
        assert provider.call_count == 1
        assert chunk.embedding is None


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_context_words_in_programming_error_do_not_trigger_repair(error):
    create = Mock(side_effect=error("maximum input length of 10000000 tokens"))
    with pytest.raises(error):
        local._embed_local_with_context_retry(SimpleNamespace(embeddings=SimpleNamespace(create=create)), "synthetic", "x" * 10000)
    assert create.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 503])
def test_provider_http_status_classification(monkeypatch, status):
    from openai import APIStatusError
    exc = APIStatusError("private", response=httpx.Response(status, request=httpx.Request("POST", "http://localhost/v1/embeddings")), body=None)
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "local-only")
    monkeypatch.setattr(facade, "get_embedding_result_via_local_openai", Mock(side_effect=exc))
    expected = facade.EmbeddingUpstreamUnavailableError if status in (408, 429, 500, 503) else EmbeddingContractError
    with pytest.raises(expected):
        facade.get_embedding_result("synthetic")


def test_strict_timeout_flows_through_facade_to_sdk(monkeypatch):
    monkeypatch.setattr(local, "monotonic", lambda: 1.0)
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "synthetic-revision")
    monkeypatch.setenv("APP_EMBED_DIMS", "1024")
    monkeypatch.setenv("APP_EMBED_MODEL", "synthetic")
    create = Mock(return_value=SimpleNamespace(model="synthetic", data=[SimpleNamespace(index=0, embedding=[1.0] * 1024)]))
    monkeypatch.setattr(local, "_get_local_openai_client", lambda *_: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    facade.get_strict_index_embeddings(["synthetic"], expected_model_signature=facade.strict_index_embedding_signature(), timeout=0.001)
    assert create.call_count == 1
    assert create.call_args.kwargs["timeout"] == pytest.approx(0.001)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_strict_timeout_never_calls_sdk(monkeypatch, timeout):
    factory = Mock()
    monkeypatch.setattr(local, "_get_local_openai_client", factory)
    with pytest.raises(ValueError):
        local.get_strict_indexed_embeddings_via_local_openai(["synthetic"], timeout=timeout)
    factory.assert_not_called()


def test_local_outage_with_unconfigured_optional_fallback_remains_recoverable(monkeypatch):
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "legacy-cohere")
    monkeypatch.setattr(facade, "get_embedding_results_via_local_openai", Mock(side_effect=ConnectionError("offline")))
    monkeypatch.setattr(facade.apps, "get_app_config", lambda *_: SimpleNamespace(cohere_client=None))
    with pytest.raises(facade.EmbeddingUpstreamUnavailableError):
        facade.get_embedding_results(["synthetic"])
