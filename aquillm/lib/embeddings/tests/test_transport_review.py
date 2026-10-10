"""Transport-boundary regressions from independent review; no network or sleeps."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest
import requests

from aquillm import utils as facade
from apps.knowledge_graph.retrieval import query_embedding
from lib.embeddings import local
from lib.embeddings.errors import require_transient
from lib.embeddings.utils import EmbeddingContractError


def strict_transport(monkeypatch, *, signature_seconds=0, client_seconds=0):
    monkeypatch.setenv("APP_EMBED_MODEL", "synthetic")
    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "synthetic-revision")
    monkeypatch.setenv("APP_EMBED_DIMS", "1024")
    signature = facade.strict_index_embedding_signature()
    clock = {"now": 1.0}
    monkeypatch.setattr(local, "monotonic", lambda: clock["now"], raising=False)
    monkeypatch.setattr(query_embedding, "monotonic", lambda: clock["now"])
    create = Mock(return_value=SimpleNamespace(
        model="synthetic", data=[SimpleNamespace(index=0, embedding=[1.0] * 1024)]
    ))
    def load_signature():
        clock["now"] += signature_seconds
        return signature
    def load_client(*_):
        clock["now"] += client_seconds
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(facade, "strict_index_embedding_signature", load_signature)
    monkeypatch.setattr(local, "_get_local_openai_client", load_client)
    return signature, create


@pytest.mark.parametrize("stage", ["signature", "client"])
def test_relative_budget_expired_during_setup_never_dispatches(monkeypatch, stage):
    signature, create = strict_transport(monkeypatch, **{f"{stage}_seconds": 0.020})
    with pytest.raises(TimeoutError):
        facade.get_strict_index_embeddings(["synthetic"], expected_model_signature=signature, timeout=0.001)
    create.assert_not_called()


def test_local_relative_budget_expired_during_client_setup_never_dispatches(monkeypatch):
    _, create = strict_transport(monkeypatch, client_seconds=0.020)
    with pytest.raises(TimeoutError):
        local.get_strict_indexed_embeddings_via_local_openai(["synthetic"], timeout=0.001)
    create.assert_not_called()


@pytest.mark.parametrize("stage", ["signature", "client"])
def test_query_absolute_deadline_expired_during_setup_never_dispatches(monkeypatch, stage):
    # For signature setup, the first lookup fits and the facade's lookup expires.
    seconds = 0.0006 if stage == "signature" else 0.020
    signature, create = strict_transport(monkeypatch, **{f"{stage}_seconds": seconds})
    with pytest.raises(TimeoutError):
        query_embedding.embed_unresolved_query_span(text="synthetic", expected_signature=signature, deadline=1.001)
    create.assert_not_called()


def test_query_partial_setup_consumption_reduces_sdk_timeout(monkeypatch):
    signature, create = strict_transport(monkeypatch, signature_seconds=0.0003, client_seconds=0.0002)
    query_embedding.embed_unresolved_query_span(text="synthetic", expected_signature=signature, deadline=1.001)
    assert create.call_count == 1
    assert create.call_args.kwargs["timeout"] == pytest.approx(0.0002)


@pytest.mark.parametrize("error_class", [httpx.UnsupportedProtocol, httpx.LocalProtocolError,
                                         httpx.InvalidURL, requests.exceptions.InvalidURL,
                                         requests.exceptions.InvalidSchema, requests.exceptions.MissingSchema])
@pytest.mark.parametrize("wrapped", [False, True])
def test_permanent_transport_configuration_is_contract_even_wrapped(error_class, wrapped):
    failure = error_class("synthetic configuration error")
    if wrapped:
        outer = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/embeddings"))
        outer.__cause__ = failure
        failure = outer
    with pytest.raises(EmbeddingContractError):
        require_transient(failure)


@pytest.mark.parametrize("error_class", [httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout,
                                         httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.parametrize("wrapped", [False, True])
def test_real_connection_and_timeout_failures_remain_recoverable(error_class, wrapped):
    failure = error_class("synthetic outage")
    if wrapped:
        outer = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/embeddings"))
        outer.__cause__ = failure
        failure = outer
    assert require_transient(failure) is None


def test_wrapped_protocol_failure_never_reaches_legacy_fallback(monkeypatch):
    failure = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/embeddings"))
    failure.__cause__ = httpx.UnsupportedProtocol("synthetic")
    monkeypatch.setenv("APP_EMBED_FALLBACK_POLICY", "legacy-cohere")
    monkeypatch.setattr(facade, "get_embedding_results_via_local_openai", Mock(side_effect=failure))
    fallback = Mock(return_value=[])
    monkeypatch.setattr(facade, "get_embedding_results_via_cohere", fallback)
    monkeypatch.setattr(facade.apps, "get_app_config", lambda *_: SimpleNamespace(cohere_client=object()))
    with pytest.raises(EmbeddingContractError):
        facade.get_embedding_results(["synthetic"])
    fallback.assert_not_called()


def test_sdk_wrapper_does_not_turn_programming_error_into_outage():
    failure = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/embeddings"))
    failure.__cause__ = TypeError("synthetic programming error")
    with pytest.raises(TypeError):
        require_transient(failure)
