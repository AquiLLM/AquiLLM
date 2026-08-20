from __future__ import annotations

from apps.knowledge_graph.retrieval import direct_seed_resolution
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier
from apps.knowledge_graph.tests.test_direct_seed_resolution import (
    K,
    Repository,
    _match,
    _ready,
    _settings,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1


class TextRepository(Repository):
    def __init__(self, tiers, texts):
        super().__init__(tiers)
        self.texts = texts

    def span_text(self, span):
        return self.texts[span.start]


def test_deduplicates_normalized_surface_and_keeps_highest_confidence() -> None:
    lower = QueryEntitySpanV1("model", 0, 5, 0.4)
    higher = QueryEntitySpanV1("model", 10, 15, 0.9)
    repository = TextRepository(
        {
            ("name", 10): (
                _match(
                    span=0,
                    entity=K[0],
                    component=K[1],
                    tier=DirectResolutionTier.NAME,
                    confidence=0.9,
                ),
            )
        },
        {0: "Alpha", 10: "alpha"},
    )

    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=(lower, higher),
        repository=repository,
        ready=_ready(),
        settings=_settings(),
        deadline=10.0,
    )

    assert outcome.diagnostics.deduplicated_span_count == 1
    assert outcome.matches[0].extraction_confidence == 0.9
    assert all(start == 10 for _tier, start in repository.calls)


def test_embedding_failure_preserves_exact_matches_and_disables_only_fallback(
    monkeypatch,
) -> None:
    spans = (
        QueryEntitySpanV1("model", 0, 5, 1.0),
        QueryEntitySpanV1("model", 6, 10, 1.0),
        QueryEntitySpanV1("model", 11, 16, 1.0),
    )
    repository = TextRepository(
        {
            ("name", 0): (
                _match(
                    span=0,
                    entity=K[0],
                    component=K[3],
                    tier=DirectResolutionTier.NAME,
                ),
            ),
            ("name", 11): (
                _match(
                    span=2,
                    entity=K[1],
                    component=K[4],
                    tier=DirectResolutionTier.NAME,
                ),
            ),
        },
        {0: "alpha", 6: "beta", 11: "gamma"},
    )
    monkeypatch.setattr(
        direct_seed_resolution,
        "embed_unresolved_query_span",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=spans,
        repository=repository,
        ready=_ready(),
        settings=_settings(embedding=True),
        deadline=10.0,
    )

    assert outcome.failure_reason is None
    assert tuple(row.span_index for row in outcome.matches) == (0, 2)
    assert outcome.diagnostics.embedding_attempt_count == 1
    assert ("name", 11) in repository.calls
