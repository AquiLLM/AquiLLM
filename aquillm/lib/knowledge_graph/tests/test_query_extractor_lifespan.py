"""Readiness must represent finished model inference, not module import."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from threading import Event
from types import SimpleNamespace

import pytest

from lib.knowledge_graph.query_extractor import service
from lib.knowledge_graph.tests.test_query_extractor_service import (
    DIGEST,
    _call,
    _request,
    _settings,
)
from lib.knowledge_graph.types import ExtractionBatchResult


class ControlledBackend:
    def __init__(self, *, blocked=False, failure=False, malformed=False):
        self.entered = Event()
        self.release = Event()
        self.finished = Event()
        self.failure = failure
        self.malformed = malformed
        self.calls = []
        if not blocked:
            self.release.set()

    def extract_entities_batch(self, texts, *, ontology):
        self.calls.append((texts, ontology.checksum))
        self.entered.set()
        try:
            if not self.release.wait(2):
                raise AssertionError("test did not release warmup")
            if self.failure:
                raise RuntimeError("secret-token /private/model/cache")
            if self.malformed:
                return ()
            return (ExtractionBatchResult(entities=(), relations=(), diagnostics=()),)
        finally:
            self.finished.set()


@pytest.fixture
def cold_runtime(monkeypatch):
    backend = ControlledBackend()
    runtime = service.QueryExtractorRuntime(
        _settings(), SimpleNamespace(checksum=DIGEST), backend
    )
    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service, "_ready", False, raising=False)
    monkeypatch.setattr(service, "_startup_started", False, raising=False)
    monkeypatch.setattr(service, "_startup_worker", None, raising=False)
    monkeypatch.setattr(service, "_shutdown_requested", False, raising=False)
    monkeypatch.setattr(service, "_inference_slots", asyncio.BoundedSemaphore(1))
    monkeypatch.setattr(service, "_STARTUP_TIMEOUT_SECONDS", 0.5, raising=False)
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        service, "load_query_extractor_settings", lambda _environment: runtime.settings
    )
    return runtime


@asynccontextmanager
async def lifespan(backend):
    incoming = asyncio.Queue()
    outgoing = asyncio.Queue()
    task = asyncio.create_task(
        service.app({"type": "lifespan"}, incoming.get, outgoing.put)
    )
    try:
        yield task, incoming, outgoing
    finally:
        backend.release.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        # Do not leave a native worker running across test event loops.
        await asyncio.wait_for(service._inference_slots.acquire(), 1)
        service._inference_slots.release()


async def next_message(outgoing):
    return await asyncio.wait_for(outgoing.get(), 1)


async def wait_for_backend(backend):
    async with asyncio.timeout(1):
        while not backend.entered.is_set():
            await asyncio.sleep(0.001)


async def assert_unready():
    health = await _call(path="/healthz", method="GET")
    request = await _call(
        path="/v1/extract", body=_request(), authorization=b"Bearer private-token"
    )
    assert health[0]["status"] == 503
    assert request[0]["status"] == 503
    assert request[1]["body"] == b'{"reason":"extractor_provenance"}'


@pytest.mark.asyncio
async def test_imported_app_is_unready_and_cannot_lazily_infer(cold_runtime):
    await assert_unready()
    assert cold_runtime.backend.calls == []


@pytest.mark.asyncio
async def test_startup_warmup_precedes_health_and_first_request(cold_runtime):
    backend = cold_runtime.backend
    backend.release.clear()
    async with lifespan(backend) as (_, incoming, outgoing):
        await incoming.put({"type": "lifespan.startup"})
        await wait_for_backend(backend)
        await assert_unready()
        assert outgoing.empty()
        backend.release.set()
        assert await next_message(outgoing) == {"type": "lifespan.startup.complete"}
        assert (await _call(path="/healthz", method="GET"))[0]["status"] == 200
        result = await _call(
            path="/v1/extract", body=_request(), authorization=b"Bearer private-token"
        )
        assert result[0]["status"] == 200
        assert backend.calls == [(("Warmup.",), DIGEST), (("A\U0001f600B",), DIGEST)]


@pytest.mark.asyncio
async def test_duplicate_startup_does_not_run_warmup_again(cold_runtime):
    async with lifespan(cold_runtime.backend) as (_, incoming, outgoing):
        for _ in range(2):
            await incoming.put({"type": "lifespan.startup"})
            assert await next_message(outgoing) == {"type": "lifespan.startup.complete"}
        assert (await _call(path="/healthz", method="GET"))[0]["status"] == 200
        assert len(cold_runtime.backend.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["configuration", "runtime", "inference", "malformed"])
async def test_startup_failure_is_fixed_and_never_publishes_readiness(
    cold_runtime, monkeypatch, kind
):
    def unavailable(*_args):
        raise RuntimeError("secret-token /private/model/cache")

    if kind == "configuration":
        monkeypatch.setattr(service, "load_query_extractor_settings", unavailable)
    elif kind == "runtime":
        monkeypatch.setattr(service, "_get_runtime", unavailable)
    else:
        setattr(cold_runtime.backend, "failure" if kind == "inference" else kind, True)
    async with lifespan(cold_runtime.backend) as (_, incoming, outgoing):
        for _ in range(2):
            await incoming.put({"type": "lifespan.startup"})
            failed = await next_message(outgoing)
            assert failed == {
                "type": "lifespan.startup.failed",
                "message": "extractor startup failed",
            }
            assert "private" not in repr(failed)
            await assert_unready()
        assert len(cold_runtime.backend.calls) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["runtime", "inference"])
async def test_timeout_holds_slot_and_late_worker_cannot_make_app_ready(
    cold_runtime, monkeypatch, phase
):
    backend = cold_runtime.backend
    backend.release.clear()
    monkeypatch.setattr(service, "_STARTUP_TIMEOUT_SECONDS", 0.02, raising=False)
    if phase == "runtime":

        def load_runtime(*_args):
            backend.entered.set()
            if not backend.release.wait(2):
                raise AssertionError("test did not release runtime initialization")
            return cold_runtime

        monkeypatch.setattr(service, "_get_runtime", load_runtime)
    async with lifespan(backend) as (_, incoming, outgoing):
        await incoming.put({"type": "lifespan.startup"})
        await wait_for_backend(backend)
        assert (await next_message(outgoing))["type"] == "lifespan.startup.failed"
        assert service._inference_slots.locked()
        await assert_unready()
        await incoming.put({"type": "lifespan.startup"})
        assert (await next_message(outgoing))["type"] == "lifespan.startup.failed"
        assert len(backend.calls) == (0 if phase == "runtime" else 1)
        backend.release.set()
        await asyncio.wait_for(service._inference_slots.acquire(), 1)
        service._inference_slots.release()
        assert backend.finished.is_set()
        await assert_unready()


@pytest.mark.asyncio
async def test_cancelled_startup_never_readies_or_starts_a_replacement(cold_runtime):
    backend = cold_runtime.backend
    backend.release.clear()
    async with lifespan(backend) as (task, incoming, outgoing):
        await incoming.put({"type": "lifespan.startup"})
        await wait_for_backend(backend)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert outgoing.empty()
        assert service._inference_slots.locked()
        async with lifespan(backend) as (_, retry, retried):
            await retry.put({"type": "lifespan.startup"})
            assert (await next_message(retried))["type"] == "lifespan.startup.failed"
            assert len(backend.calls) == 1
        await assert_unready()
        assert backend.finished.is_set()
        assert service._runtime is None
        assert service._startup_worker is None


@pytest.mark.asyncio
async def test_shutdown_clears_readiness_and_runtime(cold_runtime):
    async with lifespan(cold_runtime.backend) as (task, incoming, outgoing):
        await incoming.put({"type": "lifespan.startup"})
        assert (await next_message(outgoing))["type"] == "lifespan.startup.complete"
        assert service._runtime is cold_runtime
        await incoming.put({"type": "lifespan.shutdown"})
        assert await next_message(outgoing) == {"type": "lifespan.shutdown.complete"}
        await task
        await assert_unready()
        assert service._runtime is None
        assert service._startup_worker is None


@pytest.mark.asyncio
async def test_unready_app_still_enforces_auth_body_and_provenance(cold_runtime):
    unauthorized = await _call(path="/v1/extract", body=_request())
    assert unauthorized[0]["status"] == 401
    oversized = await _call(
        path="/v1/extract",
        body=b"private" * _settings().max_request_body_bytes,
        authorization=b"Bearer private-token",
    )
    assert oversized[0]["status"] == 413
    malformed = await _call(
        path="/v1/extract", body=b"{}", authorization=b"Bearer private-token"
    )
    assert malformed[0]["status"] == 422
    assert cold_runtime.backend.calls == []
