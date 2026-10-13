"""Healthy refresh preserves held proof; all destructive fences still revoke."""

import asyncio
import hashlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from apps.documents.services import chunk_rerank_pair_capability as capability
from apps.documents.services import pair_worker_lifecycle as worker
from apps.documents.services import rerank_transport_policy as protocol

from .test_chunk_rerank_pair_capability import Tokenizer


@pytest.fixture
def refresh(monkeypatch):
    from apps.documents.services import rag_cache

    for name in ("set_cached_rerank_capability", "delete_cached_rerank_capability"):
        monkeypatch.setattr(rag_cache, name, lambda *a, **k: None)
    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *a, **k: None)
    provider = SimpleNamespace(
        scorer_fingerprint="refresh-test",
        endpoint="http://test/score",
        shape="score_single_text_pair",
        model_name="model",
        headers={},
    )
    identity = {
        "runtime_digest": "sha256:" + "d" * 64,
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "code_revision": "c" * 40,
        "template_sha256": hashlib.sha256(b"template").hexdigest(),
    }

    def counter():
        return capability.VerifiedPairCounter(
            Tokenizer(),
            "template",
            hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            identity["tokenizer_revision"],
            identity["template_sha256"],
            time.monotonic() + 90,
        )

    old = counter()
    capability._registered[provider.scorer_fingerprint] = old
    args = dict(
        provider=provider,
        identity=identity,
        template="template",
        tokenizer_name="synthetic",
        attestation_expires=time.time() + 240,
    )
    started, release = threading.Event(), threading.Event()
    candidates = []

    def initialize(**kwargs):
        started.set()
        release.wait(2)
        result = counter()
        candidates.append(result)
        return result

    current = [args]
    ctl = worker.PairCapabilityController(
        load=lambda: dict(current[0]) if current[0] else None,
        initialize=initialize,
        timeout=0.5,
    )
    ctl.identity = dict(identity)
    state = SimpleNamespace(
        old=old,
        ctl=ctl,
        current=current,
        args=args,
        started=started,
        release=release,
        candidates=candidates,
        provider=provider,
    )
    yield state
    release.set()
    if ctl.thread:
        ctl.thread.join(1)
    ctl.invalidate()
    protocol._BINDINGS.clear()


async def pending(state):
    task = asyncio.create_task(state.ctl.attempt())
    async with asyncio.timeout(1):
        while not state.started.is_set():
            await asyncio.sleep(0.001)
    return task


@pytest.mark.asyncio
async def test_refresh_preserves_held_counter_and_revokes_all_generations(refresh):
    old, ctl = refresh.old, refresh.ctl
    expiry = old.expires
    task = await pending(refresh)
    try:
        assert old("q", "d") == 9
        assert capability.registered_pair_counter(refresh.provider) is old
    finally:
        refresh.release.set()
    new = await task
    assert new is not None and new is not old
    assert capability.registered_pair_counter(refresh.provider) is new
    assert old("q", "d") == new("q", "d") == 9
    assert old.expires == expiry
    ctl.invalidate()
    assert old("q", "d") is new("q", "d") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["withdraw", "identity", "expired", "provider"])
async def test_attestation_change_during_probe_rejects_candidate(refresh, change):
    task = await pending(refresh)
    if change == "withdraw":
        refresh.current[0] = None
    elif change == "identity":
        refresh.current[0] = dict(
            refresh.args,
            identity={
                **refresh.args["identity"],
                "runtime_digest": "sha256:" + "e" * 64,
            },
        )
    elif change == "provider":
        refresh.current[0] = dict(
            refresh.args,
            provider=SimpleNamespace(
                scorer_fingerprint="other-provider",
            ),
        )
    else:
        refresh.current[0] = dict(refresh.args, attestation_expires=time.time() - 1)
    refresh.release.set()
    assert await task is None
    assert capability.registered_pair_counter(refresh.provider) is None
    assert refresh.old("q", "d") is None


@pytest.mark.asyncio
async def test_inflight_original_expiry_is_not_extended_by_refresh(refresh):
    task = await pending(refresh)
    object.__setattr__(refresh.old, "expires", time.monotonic() - 1)
    refresh.release.set()
    new = await task
    assert new is not None and new("q", "d") == 9
    assert refresh.old("q", "d") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["invalidate", "cancel", "shutdown", "timeout"])
async def test_late_completion_cannot_resurrect_revoked_proof(refresh, fence):
    task = await pending(refresh)
    if fence == "invalidate":
        refresh.ctl.invalidate()
    elif fence == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif fence == "shutdown":
        await refresh.ctl.stop()
    else:
        assert await task is None
    assert refresh.old("q", "d") is None
    refresh.release.set()
    if fence != "cancel":
        assert await task is None
    await asyncio.to_thread(refresh.ctl.thread.join, 1)
    assert capability.registered_pair_counter(refresh.provider) is None


@pytest.mark.asyncio
async def test_failed_probe_still_revokes_held_counter(refresh):
    refresh.ctl.initialize = lambda **kwargs: None
    assert await refresh.ctl.attempt() is None
    assert refresh.old("q", "d") is None


@pytest.mark.asyncio
async def test_competing_registry_replacement_is_not_overwritten(refresh):
    task = await pending(refresh)
    other = capability.VerifiedPairCounter(None, "", "other", "t", "h", 0)
    capability.invalidate_pair_capability(refresh.provider)
    capability._registered[refresh.provider.scorer_fingerprint] = other
    refresh.release.set()
    assert await task is None
    assert refresh.old("q", "d") is None
    assert not any(
        value is refresh.candidates[-1] for value in capability._registered.values()
    )


@pytest.mark.asyncio
async def test_real_scheduled_renewal_preserves_request_reference(refresh, monkeypatch):
    clock = [time.monotonic()]

    async def sleep(_delay):
        clock[0] += 31 if clock[0] < refresh.old.expires - 60 else 0
        await asyncio.sleep(0.001)

    monkeypatch.setattr(worker, "enabled", lambda: True)
    monkeypatch.setattr(worker, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        worker,
        "asyncio",
        SimpleNamespace(
            sleep=sleep,
            CancelledError=asyncio.CancelledError,
            gather=asyncio.gather,
        ),
    )
    ctl = refresh.ctl
    ctl.task = asyncio.create_task(ctl.run(initial=refresh.old, attempted=True))
    try:
        async with asyncio.timeout(1):
            while not refresh.started.is_set():
                await asyncio.sleep(0.001)
        assert refresh.old("q", "d") == 9
    finally:
        refresh.release.set()
        await ctl.stop()


@pytest.mark.asyncio
async def test_real_three_probe_preparation_does_not_publish_early(
    refresh, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        ),
    )
    calls = []

    def post(endpoint, *, json, **kwargs):
        calls.append(endpoint)
        assert refresh.old("q", "d") == 9
        assert capability.registered_pair_counter(refresh.provider) is refresh.old
        if endpoint.endswith("/tokenize"):
            ids = Tokenizer().encode(json["prompt"], add_special_tokens=True)
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "tokens": ids,
                    "count": len(ids),
                    "max_model_len": 1024,
                },
            )
        size = len(
            Tokenizer().encode(
                "Q:" + json["text_1"] + " D:" + json["text_2"],
                add_special_tokens=True,
            )
        )
        if size > 1024:
            return SimpleNamespace(
                status_code=400,
                json=lambda: {
                    "error": {
                        "type": "BadRequestError",
                        "code": 400,
                        "param": "input_tokens",
                        "message": "context limit 1024 tokens",
                    }
                },
            )
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "score": 0.5,
                "usage": {"prompt_tokens": size},
            },
        )

    probe = capability._probe_pair_capability
    monkeypatch.setattr(
        capability, "_probe_pair_capability", lambda *a, **k: probe(*a, **k, post=post)
    )
    refresh.ctl.initialize = capability._prepare_worker_pair_capability
    new = await refresh.ctl.attempt()
    assert new is not None and new is not refresh.old
    assert refresh.ctl.budget.pairs_used == {"acquisition": 3, "final": 0}
    assert len(calls) == 6
    assert new("q", "d") == refresh.old("q", "d") == 9
