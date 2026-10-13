"""Synthetic model I/O; real gateway transport and capability probes."""
from copy import deepcopy
from hashlib import sha256
import json
from time import monotonic
from types import SimpleNamespace

import pytest
import requests

from apps.chat.tests.rag_query_treatment_transport import QueryProvider
from apps.chat.tests.rag_suffix_test_support import SyntheticTokenizer
from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services.pair_worker_lifecycle import canonical_provider
from apps.documents.services.rerank_transport_policy import (
    discover_rerank_protocol, revoke_rerank_protocol,
)
from lib.retrieval import TurnBudget, TurnLimits

BASE, MODEL = "http://p1-graph-score:8000/v1", "p1-graph-synthetic-score"


class GraphRouteProvider(QueryProvider):
    def __init__(self):
        super().__init__(rewrite=None)

    async def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "Assess evidence coverage, not relevance" not in system:
            return await super().create(**kwargs)
        self.requests.append(deepcopy(kwargs))
        self.plans.append(deepcopy(kwargs))
        text = json.dumps({"requested_aspects": ["reported use"], "support": [],
            "unresolved_aspects": ["reported use"], "next_action": None})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[], content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))


@pytest.fixture
def graph_route_score_transport(monkeypatch, settings):
    tokenizer, calls, probes = SyntheticTokenizer(), [], []
    state = SimpleNamespace(calls=calls, probes=probes, score_document=lambda text: 0.9)
    original = requests.sessions.Session.send
    for name, value in {"APP_RERANK_PROVIDER": "local", "APP_RERANK_BASE_URL": BASE,
        "APP_RERANK_MODEL": MODEL, "APP_RERANK_MODEL_REVISION": "b" * 40}.items():
        monkeypatch.setenv(name, value)
    settings.RAG_CACHE_ENABLED = False

    def send(session, request, **kwargs):
        if request.url not in {BASE.removesuffix("/v1") + "/score",
                               BASE.removesuffix("/v1") + "/tokenize"}:
            return original(session, request, **kwargs)
        payload = json.loads(request.body)
        assert payload["model"] == MODEL
        response = requests.Response()
        response.status_code, response.url, response.request = 200, request.url, request
        if request.url.endswith("/tokenize"):
            ids = tokenizer.encode(payload["prompt"])
            body = {"tokens": ids, "count": len(ids), "max_model_len": 1024}
        else:
            calls.append(deepcopy(payload))
            docs = payload["text_2"]
            docs = docs if isinstance(docs, list) else [docs]
            queries = payload["text_1"]
            queries = queries if isinstance(queries, list) else [queries] * len(docs)
            counts = [len(tokenizer.encode(q + "\n" + d)) for q, d in zip(queries, docs)]
            assert len(queries) == len(docs) and 1 <= len(docs) <= 6
            if max(counts) > 1024:
                response.status_code = 400
                body = {"error": {"type": "BadRequestError", "code": 400,
                    "param": "input_tokens", "message": "1024 context token limit"}}
            else:
                body = {"model": MODEL, "data": [
                    {"index": i, "score": state.score_document(text)} for i, text in enumerate(docs)],
                    "usage": {"prompt_tokens": sum(counts)}}
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "send", send)
    binding = discover_rerank_protocol(BASE, MODEL, deadline=monotonic() + 10)
    assert binding is not None and calls
    provider = canonical_provider(deadline=monotonic() + 15)
    budget = TurnBudget(TurnLimits())
    try:
        counter = capability.verify_pair_capability(provider, tokenizer=tokenizer,
            template="synthetic-template", identity={
                "runtime_digest": "sha256:" + "a" * 64,
                "model_revision": "b" * 40, "tokenizer_revision": "c" * 40,
                "code_revision": "d" * 40,
                "template_sha256": sha256(b"synthetic-template").hexdigest(),
            }, budget=budget)
        assert counter is not None
        probes.extend(calls)
        calls.clear()
        state.counter = counter
        yield state
    finally:
        budget.close("fixture_finished")
        capability.invalidate_pair_capability()
        revoke_rerank_protocol(BASE, binding)
