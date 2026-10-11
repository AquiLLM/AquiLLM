"""UNEXECUTED: public competing sources, real corpus producers, external scores."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from apps.chat.tests.rag_query_treatment_transport import selected_rows
from apps.documents.models import EmbeddingGeneration
from .cross_lane_graph_route_support import configure_graph_route
from .cross_lane_graph_route_transport import GraphRouteProvider

SEED = "doi:10.1000/p1-seed"
QUESTION = (f'For {SEED}, compare "P1-EXACT-Q9", terminal evidence '
            'and the final signed reading.')
EXACT = (f'P1-EXACT-Q9 is the registered code for {SEED}. '
         'doi:10.1000/island-a cites doi:10.1000/island-b.')
GRAPH = ('doi:10.1000/p1-bridge cites doi:10.1000/p1-answer. '
         'P1-GRAPH-Q9 records the terminal observation as amber.')
SIGNED = 'P1-LATE-Q9 records 7 ug/L only at pH 6.'
OPPOSITION = 'The draft 9 ug/L is contradicted; readings above pH 6 were excluded.'
FILLER = 'The final signed reading is pending; this prefix reports no result. '
LATE = (f'{SEED}. ' + FILLER * 28 + SIGNED + ' ' + OPPOSITION)


def marker_score(text):
    """A declared synthetic competition, never a support or confidence label."""
    return max((value for marker, value in (
        ('P1-EXACT-Q9', 30.0), ('P1-GRAPH-Q9', 29.0), ('P1-LATE-Q9', 28.0),
    ) if marker in text), default=0.1)


class CompetingProvider(GraphRouteProvider):
    async def create(self, **kwargs):
        if 'Assess evidence coverage, not relevance' in kwargs['messages'][0]['content']:
            return await super().create(**kwargs)
        self.requests.append(deepcopy(kwargs))
        self.answers.append(deepcopy(kwargs))
        rows = selected_rows(kwargs)
        assert 0 < len(rows) <= 3
        text = '\n'.join(row['text'].replace('. ', '. ' + row['citation'] + ' ')
                         + ' ' + row['citation'] for row in rows)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[], content=text), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))


@pytest.fixture
def competing_corpus(graph_corpus, transport, graph_route_score_transport,
                     monkeypatch, settings, projection_roles):
    configure_graph_route(monkeypatch, settings, transport[0])
    monkeypatch.setenv('RAG_DIRECT_TOP_K', '3')
    graph_route_score_transport.score_document = marker_score
    texts = {'exact': EXACT, 'seed': f'{SEED} cites doi:10.1000/p1-bridge.',
             'graph': GRAPH, 'late': LATE}
    # Five one-term distractors plus seed/late fit the unchanged exact cap.
    # All 28 still compete through real lexical similarity, never rank mocks.
    for index in range(28):
        seed = SEED if index < 5 else 'the selected source'
        texts[f'distractor-{index}'] = (
            f'For {seed}, compare terminal evidence and the final signed reading. '
            f'doi:10.1000/noise-{index}-a cites doi:10.1000/noise-{index}-b.')
    for name, text in texts.items():
        graph_corpus.add(name, collection='selected', text=text)
    graph_corpus.prepare('selected')
    projections = graph_corpus.project('selected')
    assert len(projections) == 1 and len(graph_corpus.sources) == 32
    chunks = {name: source.document.chunks.get() for name, source in graph_corpus.sources.items()}
    assert all(chunk.content == texts[name] for name, chunk in chunks.items())
    assert all(chunk.representation_generation_id is not None for chunk in chunks.values())
    assert 1600 < chunks['late'].content.index(SIGNED) < 2048
    assert len(chunks['late'].content) <= 2048
    assert not EmbeddingGeneration.objects.exists(), 'dense absence must be genuine, not a transport failure'
    return SimpleNamespace(corpus=graph_corpus, projection=projections[0], chunks=chunks, texts=texts)
