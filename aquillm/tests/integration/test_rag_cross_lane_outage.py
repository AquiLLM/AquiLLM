"""Three actual concurrent searches retain one outage latch and source authority."""
import asyncio
import json

import pytest
from django.db import connections

from apps.chat.services.rag_turn import preservation_turn
from apps.chat.tests.rag_query_treatment_support import (
    controls, query_corpus as query_corpus, close_query_worker as close_query_worker,
)
from apps.chat.tests.rag_query_treatment_observation import new_consumer, publish
from apps.chat.tests.rag_query_treatment_transport import QueryProvider, selected_rows
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.turn_context import current_turn

from .cross_lane_outage_execution import (
    QUERIES, concurrent_searches, observed_delivery, observers, synthesize,
)
from .cross_lane_outage_transport import outage_transport

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_mode", [False, True])
async def test_actual_concurrent_queries_share_outage_through_sdk_and_sinks(
    query_corpus, monkeypatch, source_mode,
):
    corpus = query_corpus
    controls(monkeypatch, "original", False)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    consumer, provider = new_consumer(corpus, QUERIES[0]), QueryProvider("unused")
    audit, score = observers(monkeypatch)
    trace, dispatches, observation = observed_delivery(corpus)
    with outage_transport(monkeypatch) as server, observation as events:
        async with preservation_turn(consumer, max_func_calls=3):
            turn, runtime = current_turn(), media_runtime()
            frozen, authority = runtime._frozen_retrieval_scope.representation, runtime.authorization
            results = await concurrent_searches(consumer, server, turn.budget)
            assert turn.budget.actions_used == 3
            assert len(audit.states) == len(audit.scores) == len(audit.rankings) == 3
            state = audit.states[0]
            assert all(row is state and row.budget is turn.budget for row in audit.states)
            assert not state.can_admit() and turn.budget.can_publish()
            for args, kwargs, scored, actual in audit.scores:
                assert actual is state
                assert scored.score_set.status == "unavailable" and scored.score_set.scores == ()
                assert scored.ranked_ids == tuple(c.pk for c in args[2])[:1]
            for _, _, ranking in audit.rankings:
                assert ranking.rerank_status == "unavailable"
                assert tuple(r.pk for r in ranking.ranked_results) == tuple(r.pk for r in ranking.combined_candidates)[:1]
            assert all(result["result"] for result in results)
            assert len({result["result"][0]["chunk_id"] for result in results}) == 1
            requests = len(server.requests)
            def repeat_failed_adapter():
                try:
                    args, kwargs, _, _ = audit.scores[0]
                    return score(*args, **kwargs)
                finally:
                    connections.close_all()
            again = await asyncio.to_thread(repeat_failed_adapter)
            assert again.score_set.status == "unavailable" and not again.score_set.scores
            assert len(server.requests) == requests == 3
            answer = await synthesize(consumer, provider, results, audit, source_mode=source_mode)
            assert len(provider.answers) == len(dispatches) == 1 and not provider.plans
            actual, lease, packet, payload = dispatches[0]
            assert actual is runtime and lease.budget is turn.budget
            assert packet.source_authorization is authority and packet._publication_representation is frozen
            selected = selected_rows(payload)
            assert selected == json.loads(lease.payload)["result"]
            assert selected and all(row["text"] == next(c.content for c in corpus.chunks
                if c.pk == row["chunk_id"]) for row in selected)
            assert packet.answer_support.state == "unknown"
            assert packet.answer_support.assessment_basis == "unvalidated"
            assert packet.answer_support.calibration_id is None and not packet.answer_support.exhaustive
            assert answer._validated_answer.outcome == "partial"
            stored, delivered = await publish(corpus, consumer)
            assert "only below 1 Pa" in stored
            assert len(delivered) == 2
            assert len(server.requests) == 3 and turn.budget.pairs_used == {"acquisition": 6, "final": 0}
            assert runtime.authorization is authority and runtime._frozen_retrieval_scope.representation is frozen
            assert current_turn() is turn
        assert not events.failed.is_set()
    assert trace.sdk_payloads and not trace.unknown
