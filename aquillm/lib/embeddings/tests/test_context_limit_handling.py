"""Tests for embedding context limit handling and retry logic."""

from types import SimpleNamespace

import pytest

from lib.embeddings import get_embedding_via_local_openai, get_embeddings_via_local_openai


CONTEXT_LIMIT_ERROR = (
    "Error code: 400 - {'error': {'message': "
    "\"You passed 2049 input tokens and requested 0 output tokens. "
    "However, the model's context length is only 2048 tokens, "
    "resulting in a maximum input length of 2048 tokens. "
    "Please reduce the length of the input prompt. "
    "(parameter=input_tokens, value=2049)\", "
    "'type': 'BadRequestError', 'param': None, 'code': 400}}"
)


class _FakeEmbeddingsApi:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append(input)
        if isinstance(input, list):
            if any(isinstance(item, str) and len(item) > 2047 for item in input):
                raise RuntimeError(CONTEXT_LIMIT_ERROR)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[float(index), 0.0, 0.0, 0.0])
                    for index, _ in enumerate(input)
                ]
            )
        if isinstance(input, str) and len(input) > 2047:
            raise RuntimeError(CONTEXT_LIMIT_ERROR)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0, 4.0])])


class _FakeOpenAIClient:
    def __init__(self, embeddings_api):
        self.embeddings = embeddings_api


def test_get_embedding_retries_with_truncated_text(monkeypatch):
    embeddings_api = _FakeEmbeddingsApi()
    monkeypatch.setattr(
        "lib.embeddings.local._get_local_openai_client",
        lambda base_url, api_key: _FakeOpenAIClient(embeddings_api),
    )

    result = get_embedding_via_local_openai("a" * 5000)

    assert result == [1.0, 2.0, 3.0, 4.0]
    assert len(embeddings_api.calls) >= 2
    assert isinstance(embeddings_api.calls[0], str)
    assert isinstance(embeddings_api.calls[-1], str)
    assert len(embeddings_api.calls[-1]) < len(embeddings_api.calls[0])


def test_get_embeddings_falls_back_to_per_item_truncation(monkeypatch):
    embeddings_api = _FakeEmbeddingsApi()
    monkeypatch.setattr(
        "lib.embeddings.local._get_local_openai_client",
        lambda base_url, api_key: _FakeOpenAIClient(embeddings_api),
    )

    result = get_embeddings_via_local_openai(["a" * 5000, "short"])

    assert len(result) == 2
    assert all(len(embedding) == 4 for embedding in result)
