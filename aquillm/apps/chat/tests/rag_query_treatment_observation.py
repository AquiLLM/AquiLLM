"""Delegates preserve real objects and all original Trace events in order."""
import json
from dataclasses import asdict
from hashlib import sha256
from time import monotonic
from types import SimpleNamespace

from channels.db import database_sync_to_async

from apps.chat.consumers.chat_publish import end_spin_publish
from apps.chat.evals.evidence_quality_trace import Trace
from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline, retrieval_query_runtime
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services import chunk_search, embedding_index_query, chunk_rerank, chunk_rerank_local_vllm
from apps.documents.services.representation_media_acquisition import media_runtime
from aquillm.message_adapters import save_conversation_to_db
from lib.evidence_observation import observe
from lib.llm.providers.final_stream import send_final_answer_stream
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

from .rag_query_treatment_support import QUESTION



def diagnose(stage, runtime, **fields):
    """Read-only original ledger metadata; no source, payload or authority cloning."""
    budget = runtime.budget if runtime is not None else current_turn().budget
    with budget._lock:
        state = {"ledger_id": id(budget), "runtime_id": id(runtime),
            "limits": asdict(budget.limits), "actions": len(budget._actions),
            "action_digests": sorted(sha256(s.encode()).hexdigest() for s in budget._actions),
            "sources": len(budget._sources), "pairs": dict(budget._pairs),
            "text": dict(budget._text), "inflight": budget._inflight,
            "planner_calls": budget._planner_calls, "deadline": budget._deadline,
            "final_deadline": budget._final_deadline, "closed": budget._closed_reason,
            "terminal": budget._terminal_reason, "sealed": budget._synthesis is not None,
            "at": monotonic()}
    row = {"stage": stage, **state, **fields}
    return row


def install_observers(monkeypatch, *, after_first=None):
    audit = SimpleNamespace(prepared=[], searches=[], rankings=[], dense=[], packets=[],
        acquisitions=[], scores=[], validations=[], validation_errors=[], mutations=[], workers=[])
    prepare = retrieval_query_runtime.prepare_query_proposals
    validate = retrieval_query_runtime.validate_bound
    search, synthesize = rag_pipeline._run_vector_search, rag_pipeline.synthesize_from_evidence
    rank, dense = chunk_search.materialize_and_rerank_candidates, embedding_index_query.rank_compatible_chunks
    score = chunk_rerank_local_vllm.rerank_via_local_vllm_scored

    async def prepared(question, **kwargs):
        result = await prepare(question, **kwargs)
        audit.prepared.append((question, kwargs, result))
        return result

    def validated(bound):
        try:
            result = validate(bound)
        except Exception as exc:
            audit.validation_errors.append((bound, bound.runtime.budget, exc))
            raise
        audit.validations.append((bound, bound.runtime.budget, result))
        return result

    def searched(consumer, query, top_k):
        runtime = media_runtime()
        started = monotonic()
        identity = {"query_sha256": sha256(query.encode()).hexdigest(), "top_k": top_k}
        audit.workers.append(diagnose("worker_entry", runtime, **identity))
        try:
            result = search(consumer, query, top_k)
        except BaseException as exc:
            audit.workers.append(diagnose("worker_error", runtime, **identity,
                elapsed_ms=(monotonic()-started)*1000, exception_type=type(exc).__name__))
            raise
        audit.workers.append(diagnose("worker_return", runtime, **identity,
            elapsed_ms=(monotonic()-started)*1000, rows=len(result.get("result", ())),
            result_status=result.get("retrieval_status")))
        audit.searches.append((query, runtime, runtime.budget, runtime.authorization, result))
        if len(audit.searches) == 1 and after_first is not None:
            before = runtime.budget.can_start_optional(3000, 1000)
            after_first()
            audit.mutations.append((before, runtime.budget.can_start_optional(3000, 1000)))
        return result

    def ranked(*args, **kwargs):
        result = rank(*args, **kwargs)
        audit.rankings.append((args, kwargs, result))
        return result

    def dense_ranked(**kwargs):
        result = dense(**kwargs)
        audit.dense.append((kwargs, result))
        return result

    def scored(*args, **kwargs):
        result = score(*args, **kwargs)
        audit.scores.append((args, kwargs, result))
        return result

    async def synthesis(llm, conversation, packet, **kwargs):
        audit.packets.append(packet)
        runtime = media_runtime()
        audit.acquisitions.append(runtime.observation.get("acquisition"))
        return await synthesize(llm, conversation, packet, **kwargs)

    monkeypatch.setattr(retrieval_query_runtime, "prepare_query_proposals", prepared)
    monkeypatch.setattr(retrieval_query_runtime, "validate_bound", validated)
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", searched)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesis)
    monkeypatch.setattr(chunk_search, "materialize_and_rerank_candidates", ranked)
    monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", dense_ranked)
    monkeypatch.setattr(chunk_rerank_local_vllm, "rerank_via_local_vllm_scored", scored)
    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm_scored", scored)
    return audit


def new_consumer(corpus, question=QUESTION, *, history=()):
    return SimpleNamespace(user=corpus.user, col_ref=CollectionsRef([corpus.collection.pk]),
        convo=Conversation(system="Use only the selected source.",
            messages=[*history, UserMessage(content=question)]))


async def publish(corpus, consumer):
    message, events = consumer.convo[-1], []
    async def stream(payload):
        events.append(("stream", payload["content"]))
    async def send(**kwargs):
        payload = json.loads(kwargs["text_data"])
        events.append(("refresh", payload["delta"]["messages"][0]["content"]))
    await send_final_answer_stream(stream, message_uuid=str(message.message_uuid),
        content=message.content, stop_reason=message.stop_reason, usage=message.usage, message=message)
    await end_spin_publish(SimpleNamespace(convo=consumer.convo,
        _spin_epoch=len(consumer.convo)-1, send=send), consumer.convo)
    await database_sync_to_async(save_conversation_to_db)(consumer.convo, corpus.db)
    stored = await database_sync_to_async(lambda: corpus.db.db_messages.get(
        message_uuid=message.message_uuid).content)()
    assert events == [("stream", message.content), ("refresh", message.content)]
    assert stored == message.content
    return stored, events


async def run(corpus, provider, consumer, *, repeat_preparation=False):
    sources = {(str(doc.id), chunk.pk): {
        "source_id": f"public-{index}", "revision": str(chunk.representation_generation_id),
        "text": chunk.content,
    } for index, (doc, chunk) in enumerate(zip(corpus.docs, corpus.chunks))}
    trace = Trace(sources)
    dispatches = []
    def sink(event, data):
        trace.sink(event, data)
        if event == "sdk_start":
            dispatches.append((media_runtime(), current_turn().budget._synthesis,
                               trace.delivered_packet, data["payload"]))
    with observe(sink) as observation:
        async with preservation_turn(consumer, max_func_calls=3):
            turn, runtime = current_turn(), media_runtime()
            diagnose("turn_ready", runtime)
            frozen, authorization = runtime._frozen_retrieval_scope.representation, runtime.authorization
            assert await rag_pipeline.run_direct_rag_turn(consumer, provider.llm, consumer.convo) == "handled"
            diagnose("direct_return", runtime)
            answer = consumer.convo[-1]
            assert answer._validated_answer is not None
            if repeat_preparation:
                before = (runtime.budget.actions_used, runtime.budget.planner_calls)
                rows = await retrieval_query_runtime.prepare_query_proposals(
                    consumer.convo.messages[0].content, llm=provider.llm, runtime=runtime,
                    anchors=getattr(runtime, "_query_proposal_anchors", None))
                assert rows is runtime._query_proposals
                assert before == (runtime.budget.actions_used, runtime.budget.planner_calls)
            stored, events = await publish(corpus, consumer)
            assert runtime.authorization is authorization
            assert runtime._frozen_retrieval_scope.representation is frozen
            assert current_turn() is turn and runtime.budget is turn.budget
    diagnose("turn_exited", runtime)
    assert not observation.failed.is_set()
    return SimpleNamespace(trace=trace, runtime=runtime, turn=turn, answer=answer,
        stored=stored, events=events, frozen=frozen, dispatches=dispatches)
