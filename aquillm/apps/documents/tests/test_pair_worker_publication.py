"""Publication rejects empty-registry ABA and late cache registration."""

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from apps.documents.services import chunk_rerank_config as config
from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services import rag_cache
from apps.documents.services import rerank_transport_policy as protocol
from lib.retrieval import TurnBudget, TurnLimits

from .test_pair_worker_refresh import pending
from .test_pair_worker_refresh import refresh as refresh


@pytest.mark.asyncio
async def test_empty_registry_invalidation_fences_initial_candidate(refresh):
    capability.invalidate_pair_capability(refresh.provider)
    task = await pending(refresh)
    capability.invalidate_pair_capability(refresh.provider)
    refresh.release.set()
    assert await task is None
    assert capability.registered_pair_counter(refresh.provider) is None


@pytest.mark.asyncio
async def test_shutdown_during_candidate_preparation_leaves_no_late_protocol(
    refresh,
    monkeypatch,
):
    base = "http://test"
    monkeypatch.setattr(config, "rerank_base_url", lambda: base)
    old = protocol.RerankProtocolV1(
        refresh.provider.endpoint,
        refresh.provider.shape,
        refresh.provider.model_name,
        time.time(),
        time.time() + 90,
    )
    protocol.register_rerank_protocol(base, old)
    refresh.ctl.protocol_binding = (base, old)
    task = await pending(refresh)
    try:
        await asyncio.wait_for(refresh.ctl.stop(), 0.1)
        assert refresh.old("q", "d") is None
        assert (
            protocol.resolve_rerank_protocol(base, refresh.provider.model_name) is None
        )
    finally:
        refresh.release.set()
    assert await task is None
    assert capability.registered_pair_counter(refresh.provider) is None
    assert protocol._BINDINGS == {}


@pytest.mark.asyncio
async def test_missing_entry_attestation_revokes_without_probes(refresh):
    refresh.current[0] = None
    assert await refresh.ctl.attempt() is None
    assert not refresh.started.is_set()
    assert refresh.old("q", "d") is None


def test_registry_never_shares_revocation_across_different_identities(refresh):
    replacement = replace(refresh.old, identity="different-proof")
    result = capability._publish_pair_candidate(
        refresh.provider,
        replacement,
        expected=capability._pair_snapshot(refresh.provider),
        budget=TurnBudget(TurnLimits()),
    )
    assert result is None
    assert capability.registered_pair_counter(refresh.provider) is refresh.old


@pytest.mark.asyncio
async def test_worker_local_protocol_never_exposes_pending_cache_write(
    refresh, monkeypatch
):
    base = "http://test"
    monkeypatch.setattr(config, "rerank_base_url", lambda: base)
    old = protocol.RerankProtocolV1(
        refresh.provider.endpoint,
        refresh.provider.shape,
        refresh.provider.model_name,
        time.time(),
        time.time() + 90,
    )
    protocol.register_rerank_protocol(base, old)
    refresh.ctl.protocol_binding = (base, old)
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        release.wait(2)

    monkeypatch.setattr(rag_cache, "set_cached_rerank_capability", blocked)
    refresh.release.set()
    task = asyncio.create_task(refresh.ctl.attempt())
    try:
        async with asyncio.timeout(1):
            while not entered.is_set() and not task.done():
                await asyncio.sleep(0.001)
        if task.done():
            assert task.result() is not None
            assert (
                protocol.resolve_rerank_protocol(base, refresh.provider.model_name)
                is not old
            )
        await asyncio.wait_for(refresh.ctl.stop(), 0.1)
        assert refresh.old("q", "d") is None
        assert (
            protocol.resolve_rerank_protocol(base, refresh.provider.model_name) is None
        )
        assert not entered.is_set()  # No shared-cache publication on this path.
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_local_revocation_preserves_newer_independent_public_binding(
    refresh, monkeypatch
):
    base = "http://test"
    monkeypatch.setattr(config, "rerank_base_url", lambda: base)
    writes = []
    monkeypatch.setattr(
        rag_cache, "set_cached_rerank_capability", lambda *a: writes.append(a)
    )
    refresh.release.set()
    assert await refresh.ctl.attempt() is not None
    assert writes == []
    newer = protocol.RerankProtocolV1(
        refresh.provider.endpoint,
        refresh.provider.shape,
        refresh.provider.model_name,
        time.time(),
        time.time() + 90,
    )
    protocol.register_rerank_protocol(base, newer)
    assert writes == [(base, refresh.provider.model_name, newer.as_cache_entry())]
    await refresh.ctl.stop()
    assert protocol.resolve_rerank_protocol(base, refresh.provider.model_name) is newer
