"""UNEXECUTED: real native structure and baseline source/window SDK/sink delivery."""
from hashlib import sha256
import json

import pytest
from channels.db import database_sync_to_async

from apps.chat.evals.structural_order_observation import section_order
from apps.chat.tests.rag_query_treatment_transport import selected_rows
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .cross_lane_graph_route_support import close_graph_route_worker as close_graph_route_worker
from .cross_lane_graph_route_transport import graph_route_score_transport as graph_route_score_transport
from .cross_lane_publication_support import publish
from .structural_delivery_support import structural_turn, quote_for
from .structural_native_files import BODY, REFERENCE, HEADER, SIGNED, OPPOSITION
from .structural_native_support import native_source as native_source, detached

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,question,required', [
    ('late', 'What signed concentration and condition does CERULEAN-OTTER report?', (HEADER, SIGNED, OPPOSITION)),
    ('body', 'What signed concentration and restriction are reported in Results?', (BODY,)),
    ('body', 'Locate the reference entry containing "Synthetic comet calibration".', (REFERENCE,)),
    ('scrambled', 'What signed concentration and restriction are reported?', ('7 ug/L only at pH 6', 'measurements above pH 6 were excluded')),
    ('unknown', 'What signed concentration and restriction are reported?', (BODY,)),
])
async def test_native_baseline_delivery_retains_required_elements(
    native_source, graph_route_score_transport, kind, question, required, monkeypatch,
):
    source = await database_sync_to_async(native_source)(kind)
    if kind == 'late':
        heading = next(e for e in source.extraction.elements if e.kind == 'heading')
        body = next(e for e in source.extraction.elements if e.kind == 'paragraph')
        chunk = next(c for c in source.chunks if SIGNED in c.content)
        assert heading.end <= body.start and 1600 < chunk.content.index(SIGNED) < 2048
        assert HEADER not in chunk.content
        entries = source.generation.manifest['chunks']
        entry = next(e for e in entries if e['number'] == chunk.chunk_number)
        assert entry['source_text'] == chunk.content
        assert entry['embedding_text'].startswith('heading=' + HEADER + '\n')
    async with structural_turn(source, question, monkeypatch) as observed:
        assert not observed.audit.runtime_calls, 'this is a graph-disabled structural baseline'
        assert len(observed.audit.snapshots) == 1
        sdk_rows = selected_rows(observed.provider.answers[-1])
        packet_rows = observed.packet.chunks
        assert len(sdk_rows) == len(packet_rows) <= 3
        assert observed.scores and observed.sealed and observed.dispatches
        runtime, turn, lease, payload = observed.dispatches[-1]
        assert runtime is observed.runtime and turn is observed.turn
        assert lease.budget is turn.budget
        for runtime_at_dispatch, turn_at_dispatch, dispatch_lease, request in observed.dispatches:
            assert runtime_at_dispatch is runtime and turn_at_dispatch is turn
            assert dispatch_lease is lease
            assert selected_rows(request) == json.loads(lease.payload)['result']
            assert any(isinstance(message.get('content'), str)
                       and message['content'].endswith(lease.payload) for message in request['messages'])
        sealed_rows = json.loads(lease.payload)['result']
        by_citation = {f'[doc:{c.doc_id} chunk:{c.pk}]': c for c in source.chunks}
        for row in sdk_rows:
            actual = by_citation[row['citation']]
            assert row['text'].encode() == actual.content.encode()
            assert next(r['text'] for r in packet_rows if r['citation'] == row['citation']) == actual.content
            assert next(r['text'] for r in sealed_rows if r['citation'] == row['citation']) == actual.content
            chosen = next(c for c in observed.packet.selection.candidates if c.chunk_id == actual.pk)
            assert chosen.source_fingerprint == sha256(actual.content.encode()).hexdigest()
            matched = [(scorer, query, scores, score) for scorer, query, _, scores in observed.scores
                       for score in scores.scores if score.chunk_pk == actual.pk]
            assert matched
            scorer, query, scores, score = matched[-1]
            assert scorer.budget is turn.budget and query == question
            assert score.source_fingerprint == chosen.source_fingerprint
            assert chosen.scorer_fingerprint == scores.scorer_fingerprint
            assert score.window_coverage.is_complete()
            assert chosen.score_interpretation == 'uncalibrated'
        selected_text = ' '.join(' '.join(row['text'].split()) for row in sdk_rows)
        for literal in required:
            assert ' '.join(literal.split()) in selected_text
        stored, events = await publish(source, observed.consumer)
        for content in (stored, *(value for _, value in events)):
            for row in sdk_rows:
                assert quote_for(content, row['citation']) == ' '.join(row['text'].split())
            quoted = ' '.join(quote_for(content, row['citation']) for row in sdk_rows)
            for literal in required:
                assert ' '.join(literal.split()) in quoted
        # Actual acquisition order is observed separately from the synthetic ordering control.
        all_rows, sources = detached(source)
        by_pk = {row['chunk_id']: row for row in all_rows}
        snapshot = observed.audit.snapshots[0][2]
        actual_rows = [by_pk[row.pk] for row in snapshot.baseline_candidates]
        report = section_order(actual_rows, sources, reference_lookup=required == (REFERENCE,))
        assert report['original_order'] == [row.pk for row in snapshot.baseline_candidates]
        assert set(report['experimental_order']) == set(report['original_order'])
        assert report['runtime_verified'] is False and report['activation_eligible'] is False
        if kind in ('unknown', 'scrambled') or required == (REFERENCE,):
            assert report['experimental_order'] == report['original_order']
        assert observed.turn.budget.actions_used == 1
    result = observed.trace.result(stored)
    assert result['events'] == observed.trace.events and result['actions'] == 1
    assert sum(event['event'] == 'turn_complete' for event in result['events']) == 1
