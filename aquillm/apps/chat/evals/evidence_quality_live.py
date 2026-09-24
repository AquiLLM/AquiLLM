"""Real isolated ASGI chat turns: actual DB retrieval, reranking and answer SDK."""

import asyncio
import json
import os
from contextlib import contextmanager
from time import perf_counter
from uuid import uuid4

from .evidence_quality_eval import digest


@contextmanager
def isolated_cache():
    from django.conf import settings
    from django.test.utils import override_settings

    namespace = "evidence-quality-" + uuid4().hex
    config = dict(settings.CACHES)
    config["default"] = {**config["default"], "KEY_PREFIX": namespace}
    with override_settings(CACHES=config):
        yield namespace


@contextmanager
def mode_environment(mode):
    preservation = mode in ("preservation", "combined")
    values = {
        "RAG_EVIDENCE_SELECTION_MODE": "adaptive"
        if mode in ("selection", "combined")
        else "legacy",
        "RAG_RERANK_TEXT_MODE": "windowed" if preservation else "legacy",
        "RAG_EVIDENCE_TEXT_MODE": "source" if preservation else "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "budgeted" if preservation else "legacy",
        "RAG_DOCUMENT_HARD_CAP": "0",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": str(int(preservation)),
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": str(int(preservation)),
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
        "RAG_DIRECT_ENABLED": "1",
    }
    before = {key: os.getenv(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def run_case(case, manifest, *, mode, cache_state):
    from channels.db import database_sync_to_async as sync_to_async

    from apps.chat.consumers.chat import ChatConsumer
    from lib.evidence_observation import observe

    from .evidence_quality_seed import create_conversation, validate_case
    from .evidence_quality_trace import Trace

    user, conversation_id, sources = await sync_to_async(create_conversation)(
        case, manifest
    )
    trace = Trace(sources)
    trace.done, trace.sdk_started = asyncio.Event(), asyncio.Event()
    incoming = asyncio.Queue()
    outbound, closed_at = [], None

    class EvaluationConsumer(ChatConsumer):
        # Persist through the real transaction/cancellation fence; omit unrelated
        # memory/index background jobs for this explicitly isolated fixture.
        async def _save_conversation(self, create_memories=False):
            return await super()._save_conversation(create_memories=False)

    consumer = EvaluationConsumer()

    async def send(message):
        nonlocal closed_at
        at = perf_counter()
        outbound.append((at, message))
        if message["type"] == "websocket.send":
            payload = json.loads(message.get("text", "{}"))
            # A delta is emitted only after completion's citation validation.
            if any(
                m.get("role") == "assistant" and m.get("content")
                for m in payload.get("delta", {}).get("messages", [])
            ):
                trace.timings.setdefault("first_grounded", (at - trace.started) * 1000)

    with observe(trace.sink):
        task = asyncio.create_task(
            consumer(
                {
                    "type": "websocket",
                    "path": f"/chat/{conversation_id}",
                    "user": user,
                    "url_route": {"kwargs": {"convo_id": conversation_id}},
                },
                incoming.get,
                send,
            )
        )
        await incoming.put({"type": "websocket.connect"})
        try:
            async with asyncio.timeout(900):
                if case["scenario"] == "cancellation":
                    await trace.sdk_started.wait()
                    closed_at = perf_counter()
                    await incoming.put({"type": "websocket.disconnect", "code": 1000})
                    await task
                else:
                    done_task = asyncio.create_task(trace.done.wait())
                    ready, _ = await asyncio.wait(
                        {done_task, task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in ready and not trace.done.is_set():
                        raise RuntimeError("ASGI exited before turn completion")
                    done_task.cancel()
                    await asyncio.gather(done_task, return_exceptions=True)
        finally:
            if not task.done():
                await incoming.put({"type": "websocket.disconnect", "code": 1000})
                try:
                    await asyncio.wait_for(asyncio.shield(task), 1)
                except TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
    await sync_to_async(validate_case)(case, manifest)
    answer = ""
    if consumer.convo:
        last = consumer.convo.messages[-1]
        if last.role == "assistant" and not last.tool_call_name:
            answer = last.content or ""
    result = trace.result(answer)
    result.update(
        publication_observation_complete=True,
        late_publications=0,
        source_validation_complete=True,
    )
    if closed_at is not None:
        late = [
            m
            for at, m in outbound
            if at > closed_at
            and m["type"] == "websocket.send"
            and (
                "delta" in json.loads(m.get("text", "{}"))
                or "stream" in json.loads(m.get("text", "{}"))
            )
        ]
        result.update(closed=True, late_publications=len(late))
        result["published_after_disconnect"] = late
        result["timings_ms"]["cancellation"] = (perf_counter() - closed_at) * 1000
    from .evidence_quality_runtime import scenario_observation, snapshot

    result.update(scenario_observation(case, trace, result))
    result.update(
        mode=mode,
        backend="live",
        cache_state=cache_state,
        snapshot=snapshot(case, manifest),
    )
    return result


async def run_live(cases, args):
    import django
    from channels.db import database_sync_to_async as sync_to_async

    from apps.documents.services.pair_worker_lifecycle import (
        PairCapabilityController,
        enabled,
    )

    from .evidence_quality_seed import seed_cases
    from .evidence_quality_trace import transport_observation

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
    await sync_to_async(django.setup)()
    manifest = (
        await sync_to_async(seed_cases)(cases, args.live_manifest)
        if args.seed
        else json.loads(args.live_manifest.read_text(encoding="utf-8"))
    )
    if manifest["source_digest"] != digest([c["sources"] for c in cases]):
        raise ValueError("fixture source snapshot mismatch")
    semaphore = asyncio.Semaphore(args.concurrency)
    unattributed = []
    worker_warmup = []
    initial = None
    preconditioning = []
    with (
        isolated_cache() as namespace,
        mode_environment(args.mode) as configuration,
        transport_observation(unattributed, worker_warmup),
    ):
        controller = PairCapabilityController() if enabled() else None
        if controller:
            initial = await controller.attempt()
            controller.task = asyncio.create_task(
                controller.run(initial, attempted=True)
            )

        async def one(case):
            async with semaphore:
                try:
                    return await run_case(
                        case, manifest, mode=args.mode, cache_state=args.cache_state
                    )
                except Exception as exc:
                    return {
                        "mode": args.mode,
                        "backend": "live",
                        "answer": "",
                        "delivered": [],
                        "citations": [],
                        "error": type(exc).__name__,
                        "provenance_complete": False,
                        "snapshot": {
                            "source": digest(case["sources"]),
                            "answer": None,
                            "reranker": None,
                            "hardware": None,
                        },
                        "timings_ms": {},
                    }

        try:
            if args.cache_state == "warm":
                for case in cases:
                    before = perf_counter()
                    pre = await one(case)
                    preconditioning.append(
                        {
                            "case_id": case["case_id"],
                            "elapsed_ms": (perf_counter() - before) * 1000,
                            "inference_pairs": pre.get("inference_pairs"),
                            "error": pre.get("error"),
                        }
                    )
            results = await asyncio.gather(*(one(case) for case in cases))
        finally:
            if controller:
                await controller.stop()
    for result in results:
        result["dispatch_accounting_complete"] = (
            result.get("dispatch_accounting_complete") is True and not unattributed
        )
    return results, {
        "configuration": configuration,
        "concurrency": args.concurrency,
        "unattributed_rerank_http": unattributed,
        "worker_warmup_http": worker_warmup,
        "pair_capability_verified": initial is not None,
        "runtime_identity": manifest.get("runtime"),
        "runtime_verified": False,
        "corpus_human_reviewed": False,
        "completion_reserve_measured": False,
        "concurrent_load_verified": args.concurrency > 1,
        "cold_warm_verified": False,
        "cache_namespace": namespace,
        "cache_state": args.cache_state,
        "preconditioning": preconditioning,
    }
