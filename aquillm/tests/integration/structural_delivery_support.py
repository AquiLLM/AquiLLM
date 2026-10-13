"""UNEXECUTED: structural baseline source/window delivery, never offline-order serving."""
from contextlib import asynccontextmanager
from itertools import groupby
import re
from types import SimpleNamespace

from apps.chat.evals.evidence_quality_live import mode_environment
from apps.chat.tests.rag_query_treatment_transport import selected_rows
from apps.chat.evals.evidence_quality_trace import Trace, transport_observation
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.chunk_rerank_window_adapter import WindowSelectionScorer
from lib.evidence_observation import observe
from lib.llm.turn_context import current_turn
from lib.llm.providers.rag_citations import find_uncited_numeric_claims
from .cross_lane_competing_support import CompetingProvider
from .cross_lane_graph_route_observation import graph_route_observation
from .cross_lane_publication_support import consumer_for, retrieve



def cited_source(text, citation):
    """External synthetic reply: cite every source line without rewriting bytes."""
    assert citation not in text
    rendered = re.sub(r"(?<=[.!?])\s+|[\r\n]+",
                      lambda match: citation + match.group(), text) + citation
    assert rendered.replace(citation, "") == text
    assert all(citation in line for line in rendered.splitlines() if line.strip())
    return rendered


class StructuralProvider(CompetingProvider):
    async def create(self, **kwargs):
        response = await super().create(**kwargs)
        if 'Assess evidence coverage, not relevance' not in kwargs['messages'][0]['content']:
            response.choices[0].message.content = '\n'.join(
                cited_source(row['text'], row['citation']) for row in selected_rows(kwargs))
        return response


@asynccontextmanager
async def structural_turn(source, question, monkeypatch):
    provider = StructuralProvider()
    consumer = consumer_for(source, question)
    trace = Trace({(str(c.doc_id), c.pk): {'source_id': str(c.pk),
        'revision': str(c.representation_generation_id), 'text': c.content} for c in source.chunks})
    unattributed = []
    with monkeypatch.context() as probe:
        audit = graph_route_observation(probe)
        scores, sealed, dispatches = [], [], []
        original_score = WindowSelectionScorer.score_windows

        def score(self, query, chunks, **kwargs):
            result = original_score(self, query, chunks, **kwargs)
            scores.append((self, query, tuple(chunks), result))
            return result

        def sink(event, data):
            trace.sink(event, data)
            if event in ('synthesis_sealed', 'sdk_start'):
                runtime, turn = media_runtime(), current_turn()
                assert runtime is not None and runtime.budget is turn.budget
                record = (runtime, turn, runtime.budget._synthesis)
                if event == 'synthesis_sealed':
                    sealed.append(record)
                elif record[2] is not None:
                    # Coverage planning also emits sdk_start before synthesis is sealed.
                    dispatches.append((*record, data['payload']))
                else:
                    assert not sealed, 'an unleased SDK call occurred after synthesis sealing'
                    assert 'Assess evidence coverage, not relevance' in data['payload']['messages'][0]['content']

        probe.setattr(WindowSelectionScorer, 'score_windows', score)
        with mode_environment('combined'), transport_observation(unattributed), observe(sink) as observer:
            async with preservation_turn(consumer, max_func_calls=3):
                turn, runtime = current_turn(), media_runtime()
                assert runtime is not None and runtime.budget is turn.budget
                frozen = runtime._frozen_retrieval_scope
                message = await retrieve(consumer, provider)
                assert provider.answers and message._validated_answer is not None
                packet = turn.answer_publication.packet
                assert packet.source_authorization is runtime.authorization
                assert packet._publication_representation is frozen.representation
                assert trace.delivered_packet is packet
                yield SimpleNamespace(provider=provider, consumer=consumer, trace=trace,
                    audit=audit, turn=turn, runtime=runtime, packet=packet, message=message,
                    scores=scores, sealed=sealed, dispatches=dispatches)
                assert current_turn() is turn and media_runtime() is runtime
                assert runtime._frozen_retrieval_scope is frozen
                turn.budget.check_active()
    assert not observer.failed.is_set() and not unattributed


def quote_for(content, citation):
    """Recover complete blocks for one identity, including uncited original lines."""
    assert not find_uncited_numeric_claims(content)
    text = []
    for quoted, lines in groupby(content.splitlines(), lambda line: line.startswith('> ')):
        if not quoted:
            continue
        block = [line.removeprefix('> ') for line in lines]
        refs = set(re.findall(r'\[doc:[^\]\s]+\s+chunk:\d+\]', '\n'.join(block)))
        if citation in refs:
            assert refs == {citation}, 'a quote block combines different source identities'
            text.extend(line.replace(' ' + citation, '') for line in block)
    assert text, 'the requested source has no attributed quote block'
    return ' '.join(' '.join(text).split())
