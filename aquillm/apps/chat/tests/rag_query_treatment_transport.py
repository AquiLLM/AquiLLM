"""Synthetic external transports; genuine protocol discovery and SDK dispatch."""
from copy import deepcopy
import json
from time import monotonic
from types import SimpleNamespace

import pytest
import requests

from apps.documents.services.rerank_transport_policy import (
    discover_rerank_protocol, revoke_rerank_protocol,
)
from lib.llm.providers.openai import OpenAIInterface


@pytest.fixture
def score_transport(query_corpus, monkeypatch):
    calls = []
    base, model = "http://synthetic-r5-score:8000/v1", "r5-constant-score"

    def send(session, request, **kwargs):
        assert request.url == "http://synthetic-r5-score:8000/score", request.url
        payload = json.loads(request.body)
        assert payload["model"] == model
        texts = payload["text_2"]
        texts = texts if isinstance(texts, list) else [texts]
        calls.append(payload)
        response = requests.Response()
        response.status_code, response.url, response.request = 200, request.url, request
        response._content = json.dumps({"model": model, "data": [
            {"index": i, "score": 0.5} for i in range(len(texts))]}).encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    binding = discover_rerank_protocol(base, model, deadline=monotonic() + 10)
    assert binding is not None and calls
    calls.clear()
    try:
        yield calls
    finally:
        revoke_rerank_protocol(base, binding)


def selected_rows(payload):
    decoder = json.JSONDecoder()
    for message in reversed(payload["messages"]):
        text = message.get("content")
        if not isinstance(text, str):
            continue
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get("result"), list):
                return value["result"]
    raise AssertionError("actual SDK request lacks selected evidence")


class QueryProvider:
    """A deterministic extractive external response, with no quality labels."""
    def __init__(self, rewrite):
        self.rewrite, self.requests, self.plans, self.answers = rewrite, [], [], []
        self.llm = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
            chat=SimpleNamespace(completions=self)), "qwen3.5:27b")

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        system = kwargs["messages"][0]["content"]
        if "Propose one retrieval query" in system:
            self.plans.append(deepcopy(kwargs))
            text = json.dumps({"rewrite": self.rewrite})
        else:
            rows = selected_rows(kwargs)
            assert rows, "unexpected answer with no actual selected source"
            self.answers.append(deepcopy(kwargs))
            row = rows[0]
            citation = row["citation"]
            text = row["text"].replace(". ", ". " + citation + " ") + " " + citation
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[], content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))
