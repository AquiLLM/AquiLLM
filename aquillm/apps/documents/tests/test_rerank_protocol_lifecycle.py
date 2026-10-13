"""test rerank protocol lifecycle: synthetic regression coverage."""

from time import time, monotonic
import pytest
from .rerank_outage_support import server, chunks, configure


def test_discovery_only_binds_verified_model_and_finite_scores(monkeypatch):
    from apps.documents.services.rerank_transport_policy import discover_rerank_protocol
    from apps.documents.services import rag_cache

    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", lambda *a, **k: None)
    with server("success") as (endpoint, received, _):
        binding = discover_rerank_protocol(
            endpoint.removesuffix("/score"),
            "synthetic",
            deadline=monotonic() + 1,
            max_attempts=3,
        )
        assert binding.endpoint == endpoint
        assert binding.model_key == "synthetic"
        assert len(received) == 1
    with server("wrong_model") as (endpoint, received, _):
        assert (
            discover_rerank_protocol(
                endpoint.removesuffix("/score"),
                "synthetic",
                deadline=monotonic() + 1,
                max_attempts=3,
            )
            is None
        )
        assert len(received) == 1


def test_request_context_cannot_trigger_discovery():
    from apps.documents.services.rerank_transport_policy import discover_rerank_protocol
    from lib.llm.turn_context import TurnContext, bind_turn
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    with bind_turn(TurnContext(TurnBudget(TurnLimits()))):
        with pytest.raises(RuntimeError):
            discover_rerank_protocol(
                "http://127.0.0.1:1", "synthetic", deadline=monotonic() + 1
            )


def test_cache_preserves_binding_identity_and_rejects_expiration(monkeypatch):
    from apps.documents.services.rerank_transport_policy import (
        RerankProtocolV1,
        resolve_rerank_protocol,
    )
    from apps.documents.services import rag_cache
    from django.test import override_settings

    with override_settings(RAG_CACHE_ENABLED=True):
        # Cache calls are isolated to a public synthetic mapping, not shared Redis.
        values = {}
        monkeypatch.setattr(
            rag_cache, "cache_set", lambda key, value, ttl: values.update({key: value})
        )
        monkeypatch.setattr(rag_cache, "cache_get", lambda key: values.get(key))
        now = time()
        binding = RerankProtocolV1(
            "http://synthetic/score",
            "score_batch_text_pairs",
            "synthetic",
            now - 1,
            now + 60,
        )
        rag_cache.set_cached_rerank_capability(
            "http://synthetic", "synthetic", binding.as_cache_entry()
        )
        assert resolve_rerank_protocol("http://synthetic", "synthetic") == binding
        assert resolve_rerank_protocol("http://synthetic", "different-model") is None
        values[next(iter(values))]["expires_at"] = now - 1
        assert resolve_rerank_protocol("http://synthetic", "synthetic") is None


@pytest.mark.asyncio
async def test_real_lifespan_warms_protocol_outside_request_and_revokes_on_shutdown(
    monkeypatch,
):
    import asyncio
    from apps.documents.services import pair_worker_lifecycle as worker
    from apps.documents.services import rag_cache
    from apps.documents.services import chunk_rerank_config as config
    from apps.documents.services.rerank_transport_policy import resolve_rerank_protocol
    from django.test import override_settings

    monkeypatch.setattr(worker, "enabled", lambda: False)
    monkeypatch.setenv("RAG_RERANK_PROTOCOL_WARM_ENABLED", "1")
    values = {}
    monkeypatch.setattr(
        rag_cache, "cache_set", lambda key, value, ttl: values.update({key: value})
    )
    monkeypatch.setattr(rag_cache, "cache_get", lambda key: values.get(key))
    monkeypatch.setattr(
        rag_cache,
        "delete_cached_rerank_capability",
        lambda base, model: values.pop(
            rag_cache.rerank_capability_cache_key(base, model), None
        ),
    )
    with (
        override_settings(RAG_CACHE_ENABLED=True),
        server("success") as (endpoint, received, started),
    ):
        base = endpoint.removesuffix("/score")
        monkeypatch.setattr(config, "rerank_base_url", lambda: base)
        monkeypatch.setattr(config, "rerank_model", lambda: "synthetic")
        monkeypatch.setattr(config, "rerank_provider", lambda: "local")
        wrapper = worker.PairCapabilityLifespan(None)
        queue = asyncio.Queue()

        async def send(message):
            pass

        task = asyncio.create_task(wrapper({"type": "lifespan"}, queue.get, send))
        await queue.put({"type": "lifespan.startup"})
        for _ in range(100):
            if resolve_rerank_protocol(base, "synthetic"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("startup did not bind a verified synthetic protocol")
        assert len(received) == 1
        second = worker.RerankProtocolController()
        assert await second.attempt() == resolve_rerank_protocol(base, "synthetic")
        assert len(received) == 1
        await queue.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(task, 0.5)
        assert resolve_rerank_protocol(base, "synthetic") is None
        await second.stop()
