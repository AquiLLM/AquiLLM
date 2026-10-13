"""Hostile external scores and delegates over genuine scoring/publication."""
import json
from time import monotonic
from types import SimpleNamespace

import pytest
import requests
from channels.db import database_sync_to_async

from apps.chat.evals.evidence_quality_trace import Trace, transport_observation
from apps.chat.services import rag_pipeline, rag_selection_coordinator, rag_selection_scoring
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services import embedding_index_query
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.rerank_transport_policy import discover_rerank_protocol, revoke_rerank_protocol
from lib.evidence_observation import observe
from lib.llm.turn_context import current_turn
from lib.llm.types.messages import ToolMessage

from .cross_lane_publication_support import publish, retrieve
from .cross_lane_support_support import CONTRADICTORY, DISTRACTOR, EXACT, IRRELEVANT, QUESTION


@pytest.fixture
def hostile_score_transport(support_corpus, monkeypatch):
    calls = []
    base, model = "http://p1-score:8000/v1", "p1-hostile-score"
    scores = {EXACT: 1000.0, IRRELEVANT: 1000.0, CONTRADICTORY: 1000.0,
              DISTRACTOR: -1000.0, "synthetic evidence": 0.5}
    def send(session, request, **kwargs):
        assert request.url == "http://p1-score:8000/score"
        payload = json.loads(request.body)
        assert payload["model"] == model
        texts = payload["text_2"]
        texts = texts if isinstance(texts, list) else [texts]
        assert all(text in scores for text in texts), "scorer must receive complete declared source bytes"
        calls.append(payload)
        response = requests.Response()
        response.status_code, response.url, response.request = 200, request.url, request
        response._content = json.dumps({"model": model,
            "data": [{"index": i, "score": scores[text]} for i, text in enumerate(texts)]}).encode()
        return response
    monkeypatch.setattr(requests.sessions.Session, "send", send)
    binding = discover_rerank_protocol(base, model, deadline=monotonic() + 10)
    assert binding is not None and calls
    calls.clear()
    try:
        yield calls
    finally:
        revoke_rerank_protocol(base, binding)


def capture(monkeypatch):
    audit = SimpleNamespace(prepared=[], scores=[], searches=[], dense=[])
    prepare = rag_selection_coordinator.prepare_selection_candidates
    score = rag_selection_scoring.score_missing_pairs
    search = rag_pipeline._run_vector_search
    rank = embedding_index_query.rank_compatible_chunks
    def prepared(**kwargs):
        result = prepare(**kwargs)
        audit.prepared.append((kwargs, result))
        return result
    def scored(**kwargs):
        result = score(**kwargs)
        audit.scores.append((kwargs, result))
        return result
    def searched(*args):
        result = search(*args)
        audit.searches.append((args, result))
        return result
    def ranked(**kwargs):
        result = rank(**kwargs)
        audit.dense.append((kwargs, result))
        return result
    monkeypatch.setattr(rag_selection_coordinator, "prepare_selection_candidates", prepared)
    monkeypatch.setattr(rag_selection_scoring, "score_missing_pairs", scored)
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", searched)
    monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", ranked)
    return audit


async def run_observed(corpus, consumer, provider):
    chunks = (corpus.chunk, corpus.distractor_chunk)
    trace = Trace({(str(chunk.doc_id), chunk.pk): {
        "source_id": str(chunk.pk), "revision": str(chunk.representation_generation_id), "text": chunk.content,
    } for chunk in chunks})
    dispatches = []
    def sink(event, data):
        trace.sink(event, data)
        if event == "sdk_start":
            dispatches.append((current_turn().budget._synthesis, trace.delivered_packet, data["payload"]))
    unattributed = []
    with transport_observation(unattributed), observe(sink) as observation:
        async with preservation_turn(consumer, max_func_calls=3):
            turn, runtime = current_turn(), media_runtime()
            authority = runtime.authorization
            assert authority is not None
            frozen = runtime._frozen_retrieval_scope.representation
            message = await retrieve(consumer, provider)
            context = turn.answer_publication
            packet = context.packet
            assert packet.query == QUESTION
            stored, events = await publish(corpus, consumer)
            tools = [m for m in consumer.convo.messages if isinstance(m, ToolMessage)]
            assert tools
            tool_results = await database_sync_to_async(lambda: list(
                corpus.db.db_messages.filter(message_uuid__in=[m.message_uuid for m in tools])
                .order_by("sequence_number").values_list("result_dict", flat=True)))()
            assert tool_results == [m.result_dict for m in tools]
            answer = message._validated_answer
            support = packet.answer_support
            assert current_turn() is turn and runtime.budget is turn.budget
            assert runtime.authorization is authority
            assert runtime._frozen_retrieval_scope.representation is frozen
            if packet.chunks:
                assert packet.source_authorization is authority
                assert packet._publication_representation is frozen
                assert context.lease is turn.budget._synthesis
    assert not observation.failed.is_set()
    assert not unattributed
    return SimpleNamespace(message=message, packet=packet, support=support, answer=answer,
        stored=stored, events=events, runtime=runtime, turn=turn, trace=trace,
        dispatches=dispatches, tool_results=tool_results)
