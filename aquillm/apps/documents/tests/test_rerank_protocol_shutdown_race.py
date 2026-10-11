"""Real protocol registration must not survive shutdown before worker return."""

import asyncio
from threading import Event

import pytest

from .rerank_outage_support import server


async def wait_for_event(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_before_return", [False, "stop", "cancel"])
async def test_protocol_registration_shutdown_owns_real_binding(
    monkeypatch, settings, shutdown_before_return,
):
    from apps.documents.services import chunk_rerank_config as config
    from apps.documents.services import rag_cache
    from apps.documents.services import rerank_transport_policy as policy
    from apps.documents.services.pair_worker_protocol import RerankProtocolController
    from apps.documents.services.rerank_request_lifetime import TurnRerankState

    settings.RAG_CACHE_ENABLED = True
    registered, release = Event(), Event()
    observed, barrier_errors = [], []
    actual_publish = TurnRerankState.publish
    controller = RerankProtocolController()
    task = None

    with server("success") as (endpoint, received, _):
        base = endpoint.removesuffix("/score")
        key = (base, "synthetic")
        monkeypatch.setattr(config, "rerank_base_url", lambda: base)
        monkeypatch.setattr(config, "rerank_model", lambda: "synthetic")
        monkeypatch.setattr(config, "rerank_headers", lambda: {})
        # No capability/discovery/cache outcome is substituted. The observer
        # pauses only after the real publish callback and its locks return.
        def observe_publish(state, callback):
            admitted = actual_publish(state, callback)
            if admitted and state.budget is controller.budget:
                with policy._BINDING_LOCK:
                    binding = policy._BINDINGS.get(key)
                cached = rag_cache.get_cached_rerank_capability(base, "synthetic")
                observed.append((binding, cached))
                registered.set()
                if not release.wait(5):
                    barrier_errors.append("registration barrier timed out")
            return admitted

        monkeypatch.setattr(TurnRerankState, "publish", observe_publish)
        # Fresh ephemeral endpoint; reject stale/local-only prerequisites.
        assert policy.resolve_rerank_protocol(base, "synthetic") is None
        try:
            controller.task = task = asyncio.create_task(controller.attempt())
            await wait_for_event(registered)
            assert len(observed) == 1 and len(received) == 1
            binding, cached = observed[0]
            assert isinstance(binding, policy.RerankProtocolV1)
            assert binding.endpoint == endpoint and binding.model_key == "synthetic"
            assert binding.payload_kind == "score_batch_text_pairs"
            assert cached == binding.as_cache_entry()
            assert received[0] == {
                "model": "synthetic", "text_1": "protocol verification",
                "text_2": ["synthetic evidence"],
            }
            assert controller.thread.is_alive()
            assert controller.budget.can_publish()
            if shutdown_before_return:
                if shutdown_before_return == "stop":
                    await asyncio.wait_for(controller.stop(), 1)
                else:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                assert not controller.budget.can_publish()
                assert task.cancelled() and controller.thread.is_alive()
                release.set()
            else:
                release.set()
                assert await asyncio.wait_for(task, 1) is binding
                assert controller.binding is binding
                assert policy.resolve_rerank_protocol(base, "synthetic") is binding
                assert rag_cache.get_cached_rerank_capability(
                    base, "synthetic",
                ) == binding.as_cache_entry()
                await asyncio.wait_for(controller.stop(), 1)
            await asyncio.to_thread(controller.thread.join, 1)
            assert not controller.thread.is_alive() and not barrier_errors
            assert controller.binding is None
            with policy._BINDING_LOCK:
                local_after_stop = policy._BINDINGS.get(key)
            cached_after_stop = rag_cache.get_cached_rerank_capability(base, "synthetic")
            assert (local_after_stop, cached_after_stop) == (None, None)
            assert policy.resolve_rerank_protocol(base, "synthetic") is None
            assert len(received) == 1
        finally:
            release.set()
            await controller.stop()
            if controller.thread:
                await asyncio.to_thread(controller.thread.join, 1)
            # Cleanup follows all assertions, including an expected RED. It
            # cannot make the shutdown/cache checks above pass accidentally.
            for binding, _ in observed:
                if isinstance(binding, policy.RerankProtocolV1):
                    policy.revoke_rerank_protocol(base, binding)


@pytest.mark.asyncio
async def test_completed_worker_cannot_overlap_unresumed_attempt(monkeypatch, settings):
    from apps.documents.services import chunk_rerank_config as config
    from apps.documents.services import pair_worker_lifecycle as worker
    from apps.documents.services.pair_worker_protocol import RerankProtocolController

    from apps.documents.services.rerank_request_lifetime import TurnRerankState

    settings.RAG_CACHE_ENABLED = True
    allow_worker, published = Event(), Event()
    barrier_errors = []
    actual_publish = TurnRerankState.publish
    parked, resume = asyncio.Event(), asyncio.Event()
    actual_sleep = asyncio.sleep
    controller = RerankProtocolController()
    first = None

    def hold_published_worker(state, callback):
        admitted = actual_publish(state, callback)
        if admitted and state.budget is controller.budget:
            published.set()
            if not allow_worker.wait(5):
                barrier_errors.append("overlap publication barrier timed out")
        return admitted

    async def pause_before_resume(delay):
        if asyncio.current_task() is first:
            allow_worker.set()
            while controller.thread.is_alive():
                await actual_sleep(0.005)
            parked.set()
            await resume.wait()
        else:
            await actual_sleep(delay)

    with server("success") as (endpoint, received, _):
        base = endpoint.removesuffix("/score")
        monkeypatch.setattr(config, "rerank_base_url", lambda: base)
        monkeypatch.setattr(config, "rerank_model", lambda: "synthetic")
        monkeypatch.setattr(config, "rerank_headers", lambda: {})
        monkeypatch.setattr(worker.asyncio, "sleep", pause_before_resume)
        monkeypatch.setattr(TurnRerankState, "publish", hold_published_worker)
        try:
            controller.task = first = asyncio.create_task(controller.attempt())
            await asyncio.wait_for(parked.wait(), 3)
            assert not controller.thread.is_alive() and not first.done()
            assert published.is_set() and not barrier_errors
            original_budget = controller.budget
            assert original_budget.can_publish() and len(received) == 1
            assert await controller.attempt() is None
            assert controller.budget is original_budget
            resume.set()
            binding = await asyncio.wait_for(first, 1)
            assert binding is not None and controller.binding is binding
            assert len(received) == 1
        finally:
            allow_worker.set()
            resume.set()
            await controller.stop()
            if controller.thread:
                await asyncio.to_thread(controller.thread.join, 1)


@pytest.mark.asyncio
async def test_old_controller_stop_preserves_newer_real_discovery(monkeypatch, settings):
    from time import monotonic
    from apps.documents.services import chunk_rerank_config as config
    from apps.documents.services import rag_cache
    from apps.documents.services import rerank_transport_policy as policy
    from apps.documents.services.pair_worker_protocol import RerankProtocolController

    settings.RAG_CACHE_ENABLED = True
    controller = RerankProtocolController()
    newer = None
    with server("success") as (endpoint, received, _):
        base = endpoint.removesuffix("/score")
        monkeypatch.setattr(config, "rerank_base_url", lambda: base)
        monkeypatch.setattr(config, "rerank_model", lambda: "synthetic")
        monkeypatch.setattr(config, "rerank_headers", lambda: {})
        try:
            old = await controller.attempt()
            assert old is not None and controller.binding is old
            assert rag_cache.get_cached_rerank_capability(
                base, "synthetic",
            ) == old.as_cache_entry()
            newer = await asyncio.to_thread(
                policy.discover_rerank_protocol, base, "synthetic",
                deadline=monotonic() + 3, max_attempts=1,
            )
            assert newer is not None and newer != old
            assert len(received) == 2
            await controller.stop()
            assert controller.binding is None
            assert policy.resolve_rerank_protocol(base, "synthetic") is newer
            assert rag_cache.get_cached_rerank_capability(
                base, "synthetic",
            ) == newer.as_cache_entry()
        finally:
            await controller.stop()
            if controller.thread:
                await asyncio.to_thread(controller.thread.join, 1)
            if newer is not None:
                policy.revoke_rerank_protocol(base, newer)


def test_failed_ownership_observer_revokes_actual_registration(monkeypatch, settings):
    from time import monotonic
    from apps.documents.services import rag_cache
    from apps.documents.services import rerank_transport_policy as policy

    settings.RAG_CACHE_ENABLED = True
    observed, revoked = [], []
    actual_revoke = policy.revoke_rerank_protocol

    def record_revoke(base, binding):
        revoked.append((base, binding))
        return actual_revoke(base, binding)

    monkeypatch.setattr(policy, "revoke_rerank_protocol", record_revoke)
    with server("success") as (endpoint, received, _):
        base = endpoint.removesuffix("/score")

        def reject_ownership(binding):
            assert policy.resolve_rerank_protocol(base, "synthetic") is binding
            assert rag_cache.get_cached_rerank_capability(
                base, "synthetic",
            ) == binding.as_cache_entry()
            observed.append(binding)
            raise RuntimeError("synthetic ownership observer failure")

        with pytest.raises(RuntimeError, match="^synthetic ownership observer failure$"):
            policy.discover_rerank_protocol(
                base, "synthetic", deadline=monotonic() + 3, max_attempts=1,
                on_registered=reject_ownership,
            )
        assert len(received) == 1 and len(observed) == 1
        assert revoked == [(base, observed[0])]
        assert policy.resolve_rerank_protocol(base, "synthetic") is None
        assert rag_cache.get_cached_rerank_capability(base, "synthetic") is None
