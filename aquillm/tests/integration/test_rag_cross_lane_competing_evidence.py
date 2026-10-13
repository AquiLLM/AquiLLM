"""UNEXECUTED: real exact/graph/late competition; failed premises remain failures."""
from hashlib import sha256

import pytest

from apps.chat.tests.rag_query_treatment_transport import selected_rows
from apps.knowledge_graph.models import ProjectionChunkReference
from channels.db import database_sync_to_async
from lib.embeddings.tests.test_document_provider import transport as transport
from .cross_lane_competing_support import (
    competing_corpus as competing_corpus, CompetingProvider, QUESTION, SEED, SIGNED, OPPOSITION,
)
from .cross_lane_graph_route_observation import assert_graph_originals
from .cross_lane_graph_route_support import close_graph_route_worker as close_graph_route_worker, graph_route_turn
from .cross_lane_graph_route_transport import graph_route_score_transport as graph_route_score_transport
from .cross_lane_publication_support import publish
from .persisted_graph_corpus_support import graph_corpus as graph_corpus
from .persisted_graph_gateway_support import owned_gateway
from .persisted_graph_projection_support import graph_configuration as graph_configuration
from .persisted_graph_role_support import projection_roles as projection_roles
from .persisted_graph_source_support import upload as upload
from .structural_delivery_support import quote_for

pytestmark = pytest.mark.django_db(transaction=True,
    databases=['default', 'projection_source', 'projection_state'])


def scalar_pairs(calls):
    for payload in calls:
        docs = payload['text_2']
        docs = docs if isinstance(docs, list) else [docs]
        queries = payload['text_1']
        queries = queries if isinstance(queries, list) else [queries] * len(docs)
        assert len(queries) == len(docs)
        yield from zip(queries, docs)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['baseline', 'selection', 'preservation', 'combined'])
async def test_distinct_evidence_competes_in_actual_graph_route(
    competing_corpus, graph_route_score_transport, mode, monkeypatch, settings, tmp_path,
):
    fixture, provider = competing_corpus, CompetingProvider()
    expected = {fixture.chunks[name].pk for name in ('exact', 'graph', 'late')}
    mapping = await database_sync_to_async(lambda: dict(ProjectionChunkReference.objects.filter(
        projection=fixture.projection).values_list('projection_chunk_key', 'integer_chunk_pk')))()
    with owned_gateway(tmp_path) as gateway:
        async with graph_route_turn(fixture.corpus, gateway, mode=mode, question=QUESTION,
            collection_names=('selected',), query_entity=('paper', SEED), provider=provider,
            monkeypatch=monkeypatch, settings=settings) as observed:
            envelope = assert_graph_originals(observed)  # Mandatory actual nonnull G3 prerequisite.
            assert len(observed.audit.snapshots) == 1
            args, kwargs, snapshot = observed.audit.snapshots[0]
            assert args[1:3] == (QUESTION, 9)
            assert kwargs['query_embedding'].compatibility_key is not None
            assert snapshot.vector_error is None and not snapshot.vector_chunk_ids
            # Raw cap is 27; unchanged app defaults can bound effective limits to 12.
            assert 11 <= snapshot.effective_limits.exact <= 27
            assert 11 <= snapshot.effective_limits.trigram <= 27
            graph_ids = {mapping[ref.chunk_key] for ref in envelope.handoff.candidates}
            exact, graph, late = (fixture.chunks[name] for name in ('exact', 'graph', 'late'))
            assert exact.pk in snapshot.exact_chunk_ids
            assert exact.pk not in (*snapshot.trigram_chunk_ids, *snapshot.full_text_chunk_ids, *graph_ids)
            assert graph.pk in graph_ids
            assert graph.pk not in {row.pk for row in snapshot.baseline_candidates}
            assert late.pk in {row.pk for row in snapshot.baseline_candidates}
            ranking = observed.audit.rankings[-1][2]
            assert len(ranking.combined_candidates) > 9
            assert expected <= {row.pk for row in ranking.combined_candidates}
            ranked_ids = {row.pk for row in ranking.ranked_results}
            assert len(ranked_ids) == 9 and {exact.pk, graph.pk} <= ranked_ids
            if mode in ('preservation', 'combined'):
                assert expected <= ranked_ids
            assert provider.answers and observed.message._validated_answer is not None
            packet = observed.turn.answer_publication.packet
            assert packet.source_authorization is observed.runtime.authorization
            assert packet._publication_representation is observed.representation
            assert observed.trace.delivered_packet is packet
            rows = selected_rows(provider.answers[-1])
            citations = {row['citation'] for row in packet.chunks}
            assert len(citations) == 3 and {row['citation'] for row in rows} == citations
            assert {f'[doc:{row.doc_id} chunk:{row.pk}]' for row in (exact, graph)} <= citations
            if mode in ('preservation', 'combined'):
                assert citations == {f'[doc:{row.doc_id} chunk:{row.pk}]' for row in (exact, graph, late)}
            stored, events = await publish(observed.source, observed.consumer)
            for text in (stored, observed.message.content, *(value for _, value in events)):
                assert 'P1-EXACT-Q9' in text and 'P1-GRAPH-Q9' in text
                assert all(citation in text for citation in citations)
            if mode in ('preservation', 'combined'):
                scores = ranking.score_set
                assert scores is not None and scores.status == 'complete'
                score = next(row for row in scores.scores if row.chunk_pk == late.pk)
                assert score.source_fingerprint == sha256(late.content.encode()).hexdigest()
                assert score.window_coverage.is_complete()
                assert len(score.window_coverage.required_window_ids) >= 2
                pairs = list(scalar_pairs(graph_route_score_transport.calls))
                assert any(SIGNED in text and OPPOSITION in text for _, text in pairs)
                assert all(query == QUESTION and graph_route_score_transport.counter(query, text) <= 1024
                           for query, text in pairs)
                prepared_kwargs, prepared = observed.audit.prepared[-1]
                assert prepared_kwargs['turn_budget'] is observed.turn.budget
                assert prepared_kwargs['authorization'] is observed.runtime.authorization
                assert prepared.reused_pairs == len(prepared.candidates) > 0
                assert prepared.new_pairs == 0
                assert {row.chunk_id for row in prepared.candidates} <= ranked_ids
                chosen = next(row for row in packet.selection.candidates if row.chunk_id == late.pk)
                assert chosen.source_fingerprint == score.source_fingerprint
                assert chosen.scorer_fingerprint == scores.scorer_fingerprint
                assert chosen.score_interpretation == 'uncalibrated'
                for delivered in (packet.chunks, rows):
                    assert next(row['text'] for row in delivered if row['citation'] ==
                        f'[doc:{late.doc_id} chunk:{late.pk}]').encode() == late.content.encode()
                for content in (stored, *(value for _, value in events)):
                    assert content.index(SIGNED) < content.index(OPPOSITION)
                    citation = f'[doc:{late.doc_id} chunk:{late.pk}]'
                    assert quote_for(content, citation) == ' '.join(late.content.split())
            else:
                # Honest prefix controls; never label their numerical score full-source coverage.
                if ranking.score_set is not None:
                    assert all(row.window_coverage is None for row in ranking.score_set.scores)
        # Final ledger counters are emitted when preservation_turn exits.
        result = observed.trace.result(stored)
        assert result['events'] == observed.trace.events
        assert result['actions'] == 1
