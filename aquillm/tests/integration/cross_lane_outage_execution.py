"""Actual concurrent tool services, frozen selection and all final sinks."""
import asyncio
from threading import Barrier, Lock
from time import monotonic
from types import SimpleNamespace

from django.db import connections

from apps.chat.evals.evidence_quality_trace import Trace
from apps.chat.services import rag_pipeline
from apps.chat.services.rag_answer_publication import bind_legacy_answer_packet
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_pipeline_messages import _append_retrieval_messages
from apps.chat.services.rag_preservation_turn import finish_preservation
from apps.chat.services.rag_retrieval import merge_ranked_tool_results
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.documents.services import chunk_rerank, chunk_rerank_local_vllm, chunk_search
from apps.documents.services.rerank_request_lifetime import rerank_state_for
from apps.documents.services.representation_media_acquisition import media_runtime
from lib.evidence_observation import observe
from lib.llm.turn_context import current_turn
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

QUERIES = ("Explain BRX-9 response", "Describe BRX-9 response", "Summarize BRX-9 response")


def observers(monkeypatch):
    audit = SimpleNamespace(scores=[], rankings=[], states=[], packets=[], lock=Lock())
    barrier = Barrier(3, timeout=3)
    score = chunk_rerank_local_vllm.rerank_via_local_vllm_scored
    rank = chunk_search.materialize_and_rerank_candidates

    def scored(*args, **kwargs):
        runtime = media_runtime()
        state = rerank_state_for(kwargs.get("turn_budget"), kwargs.get("rerank_state"))
        assert state.budget is runtime.budget is current_turn().budget
        with audit.lock:
            audit.states.append(state)
        barrier.wait()
        result = score(*args, **kwargs)
        with audit.lock:
            audit.scores.append((args, kwargs, result, state))
        return result

    def ranked(*args, **kwargs):
        result = rank(*args, **kwargs)
        with audit.lock:
            audit.rankings.append((args, kwargs, result))
        return result

    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", scored)
    monkeypatch.setattr(chunk_rerank_local_vllm, "rerank_via_local_vllm_scored", scored)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", ranked)
    return audit, score


async def _wait_for_batches(server, tasks, budget):
    # Synchronization spans acquisition; keep a reserve in the ORIGINAL turn.
    cutoff = monotonic() + max(0, budget.remaining_ms() - 1000) / 1000
    while not server.full.is_set():
        remaining = cutoff - monotonic()
        if remaining <= 0 or not budget.can_publish() or all(t.done() for t in tasks):
            return False
        await asyncio.wait(tasks, timeout=min(0.025, remaining))
    return True


def _task_outcomes(tasks):
    outcomes = []
    for task in tasks:
        if not task.done():
            outcomes.append({"state": "pending"})
        elif task.cancelled():
            outcomes.append({"state": "cancelled"})
        elif (error := task.exception()) is not None:
            outcomes.append({"state": "error", "type": type(error).__name__})
        else:
            result = task.result()
            rows = result.get("result") if isinstance(result, dict) else None
            outcomes.append({"state": "returned", "rows": len(rows) if isinstance(rows, list) else None})
    return outcomes


async def concurrent_searches(consumer, server, budget):
    def work(query):
        try:
            return rag_pipeline._run_vector_search(consumer, query, 1)
        finally:
            connections.close_all()
    tasks = [asyncio.create_task(asyncio.to_thread(work, query)) for query in QUERIES]
    try:
        assert await _wait_for_batches(server, tasks, budget), (
            "all three real batches must reach HTTP",
            {"requests": len(server.requests), "active_pairs": server.active_pairs,
             "pairs": budget.pairs_used, "remaining_ms": budget.remaining_ms(),
             "tasks_done": [task.done() for task in tasks], "server_errors": server.errors,
             "task_outcomes": _task_outcomes(tasks)},
        )
        assert len(server.requests) == 3 and sum(len(r["text_2"]) for r in server.requests) == 6
        assert server.active_pairs == server.peak_pairs == 6
        assert budget.pairs_used == {"acquisition": 6, "final": 0}
        assert budget.can_publish() and budget.remaining_ms() >= 1000
        probe = TurnBudget(TurnLimits())
        admitted = probe.start_pair(phase="acquisition")
        try:
            assert not admitted, "real process permits must remain occupied"
        finally:
            if admitted:
                probe.finish_pair()
    finally:
        server.release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    admitted = probe.start_pair(phase="acquisition")
    try:
        assert admitted, "actual completed I/O must release permits"
    finally:
        if admitted:
            probe.finish_pair()
    assert await asyncio.to_thread(server.idle.wait, 1)
    assert server.active_pairs == 0 and not server.errors
    return results


async def synthesize(consumer, provider, results, audit, *, source_mode):
    async def observed_synthesis(llm, conversation, packet, **kwargs):
        audit.packets.append(packet)
        return await synthesize_from_evidence(llm, conversation, packet, **kwargs)

    if source_mode:
        consumer.convo = await finish_preservation(consumer, provider.llm, consumer.convo,
            question=QUERIES[0], initial_results=results, initial_acquired=True,
            iterative=False, synthesis_fn=observed_synthesis)
    else:
        raw = merge_ranked_tool_results(results, limit=1)
        packet = build_evidence_packet(raw, query=QUERIES[0], search_scope="selected documents")
        await bind_legacy_answer_packet(consumer, packet)
        conversation = _append_retrieval_messages(consumer.convo, QUERIES[0], raw, 1)
        consumer.convo = await observed_synthesis(provider.llm, conversation, packet)
    return consumer.convo[-1]


def observed_delivery(corpus):
    sources = {(str(doc.id), chunk.pk): {"source_id": f"outage-{index}",
        "revision": str(chunk.representation_generation_id), "text": chunk.content}
        for index, (doc, chunk) in enumerate(zip(corpus.docs, corpus.chunks))}
    trace, dispatches = Trace(sources), []
    def sink(event, data):
        trace.sink(event, data)
        if event == "sdk_start":
            dispatches.append((media_runtime(), current_turn().budget._synthesis,
                               trace.delivered_packet, data["payload"]))
    return trace, dispatches, observe(sink)
