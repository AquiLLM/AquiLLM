"""Serving workers alone own bounded warm attempts and invalidate on failure."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services import pair_worker_lifecycle as worker


@pytest.mark.asyncio
async def test_all_off_lifespan_has_zero_probes_and_forwards(monkeypatch):
    monkeypatch.delenv("RAG_PAIR_CAPABILITY_WARM_ENABLED", raising=False)
    called = []

    async def app(scope, receive, send):
        called.append(scope["type"])

    wrapper = worker.PairCapabilityLifespan(app)
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])
    replies = []

    async def receive():
        return next(messages)

    async def send(message):
        replies.append(message["type"])

    await wrapper({"type": "lifespan"}, receive, send)
    await wrapper({"type": "http"}, receive, send)
    assert replies == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert called == ["http"]
    assert wrapper.controller is None


@pytest.mark.asyncio
async def test_whole_tokenizer_attempt_timeout_fences_late_publication():
    release = threading.Event()
    registered = []
    provider = SimpleNamespace(scorer_fingerprint="test-worker")

    def initialize(**kwargs):
        release.wait(1)
        kwargs["budget"].publish(lambda: registered.append(True))

    ctl = worker.PairCapabilityController(
        load=lambda: {"provider": provider}, initialize=initialize, timeout=0.02
    )
    try:
        assert await ctl.attempt() is None
        assert not ctl.budget.can_publish()
        # A timed-out native call can occupy only the single warm slot.
        assert await ctl.attempt() is None
    finally:
        release.set()
        await asyncio.sleep(0.03)
        await ctl.stop()
    assert not registered


def test_explicit_invalidation_revokes_existing_counter_reference():
    counter = capability.VerifiedPairCounter(None, "", "i", "t", "h", float("inf"))
    provider = SimpleNamespace(scorer_fingerprint="invalidate-test")
    capability._registered[provider.scorer_fingerprint] = counter
    capability.invalidate_pair_capability(provider)
    assert capability.registered_pair_counter(provider) is None
    assert counter("query", "source") is None


def test_no_attestation_means_unknown(monkeypatch):
    monkeypatch.delenv("RAG_PAIR_CAPABILITY_ATTESTATION", raising=False)
    assert worker.load_attested_attempt() is None


def test_canonical_verified_provider_needs_no_shared_cache(monkeypatch):
    from apps.documents.services import chunk_rerank_selection_provider as factory

    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    provider = worker.canonical_provider(deadline=float("inf"))
    assert provider is not None
    counter = capability.VerifiedPairCounter(None, "", "i", "t", "h", float("inf"))
    capability._registered[provider.scorer_fingerprint] = counter
    monkeypatch.setattr(
        factory.rag_cache,
        "get_cached_rerank_capability",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache read")),
    )
    try:
        result = factory.current_selection_scorer(deadline=float("inf"), windowed=True)
        assert result is not None
    finally:
        capability.invalidate_pair_capability(provider)


@pytest.mark.asyncio
async def test_failed_renewal_revokes_held_counter_and_shutdown_stays_finite():
    provider = SimpleNamespace(scorer_fingerprint="renewal-test")
    available = [True]

    def load():
        return {"provider": provider} if available[0] else None

    def initialize(**kwargs):
        counter = capability.VerifiedPairCounter(None, "", "i", "t", "h", float("inf"))
        kwargs["budget"].publish(
            lambda: capability._registered.update(
                {provider.scorer_fingerprint: counter}
            )
        )
        return counter

    ctl = worker.PairCapabilityController(load=load, initialize=initialize)
    counter = await ctl.attempt()
    assert capability.registered_pair_counter(provider) is counter
    available[0] = False
    assert await ctl.attempt() is None
    assert counter("q", "d") is None
    await asyncio.wait_for(ctl.stop(), 0.1)


@pytest.mark.asyncio
async def test_renewal_starts_before_expiry_and_disabling_revokes(monkeypatch):
    import time

    active = [True]
    monkeypatch.setattr(worker, "enabled", lambda: active[0])
    ctl = worker.PairCapabilityController()
    attempts = []

    async def attempt():
        attempts.append(time.monotonic())
        if len(attempts) == 2:
            active[0] = False
        return SimpleNamespace(expires=time.monotonic() + 30)

    ctl.attempt = attempt
    ctl.task = asyncio.create_task(ctl.run())
    await asyncio.sleep(0.03)
    active[0] = False
    await ctl.stop()
    assert len(attempts) == 1  # short remaining attestation never causes a probe spin


def test_attestation_checks_pinned_identity_expiry_and_exact_endpoint(
    tmp_path, monkeypatch
):
    import json
    import time

    from apps.documents.services.chunk_rerank_results import fingerprint_text

    for key, value in {
        "APP_RERANK_PROVIDER": "local",
        "APP_RERANK_TOKENIZER": "pinned",
        "APP_RERANK_MODEL_REVISION": "a" * 40,
        "APP_RERANK_TOKENIZER_REVISION": "b" * 40,
        "APP_RERANK_CODE_REVISION": "c" * 40,
    }.items():
        monkeypatch.setenv(key, value)
    provider = worker.canonical_provider(deadline=time.monotonic() + 15)
    data = {
        "verified_by": "operator",
        "verification_record": "runtime-proof.json",
        "verified_at": time.time() - 1,
        "expires_at": time.time() + 90,
        "endpoint": provider.endpoint,
        "served_model": provider.model_name,
        "pair_context": 1024,
        "tokenizer_name": "pinned",
        "template": "template",
        "identity": {
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "code_revision": "c" * 40,
            "template_sha256": fingerprint_text("template"),
            "runtime_digest": "sha256:" + "d" * 64,
        },
    }
    path = tmp_path / "attestation.json"
    monkeypatch.setenv("RAG_PAIR_CAPABILITY_ATTESTATION", str(path))
    path.write_text(json.dumps(data))
    assert worker.load_attested_attempt() is not None
    data["expires_at"] = time.time() - 1
    path.write_text(json.dumps(data))
    assert worker.load_attested_attempt() is None
    data["expires_at"] = time.time() + 90
    data["endpoint"] += "/wrong"
    path.write_text(json.dumps(data))
    assert worker.load_attested_attempt() is None


@pytest.mark.asyncio
async def test_enabled_actual_lifespan_starts_worker_and_shutdown_revokes(monkeypatch):
    initialized = threading.Event()

    def initialize(**kwargs):
        initialized.set()
        return None

    ctl = worker.PairCapabilityController(
        load=lambda: {"provider": None}, initialize=initialize
    )
    monkeypatch.setattr(worker, "enabled", lambda: True)
    monkeypatch.setattr(worker, "PairCapabilityController", lambda: ctl)
    wrapper = worker.PairCapabilityLifespan(None)
    queue = asyncio.Queue()
    replies = []

    async def send(message):
        replies.append(message)

    task = asyncio.create_task(wrapper({"type": "lifespan"}, queue.get, send))
    await queue.put({"type": "lifespan.startup"})
    for _ in range(30):
        if initialized.is_set():
            break
        await asyncio.sleep(0.01)
    assert initialized.is_set()
    await queue.put({"type": "lifespan.shutdown"})
    await asyncio.wait_for(task, 0.5)
    assert ctl.stopping and not ctl.budget.can_publish()
    assert [r["type"] for r in replies] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


@pytest.mark.asyncio
async def test_attestation_expiring_during_warm_work_cannot_publish():
    import time

    published = []

    def initialize(**kwargs):
        time.sleep(0.03)
        kwargs["budget"].publish(lambda: published.append(True))

    ctl = worker.PairCapabilityController(
        load=lambda: {"provider": None, "attestation_expires": time.time() + 0.01},
        initialize=initialize,
        timeout=0.1,
    )
    await ctl.attempt()
    await ctl.stop()
    assert published == []
