# ruff: noqa: E501
from __future__ import annotations

from dataclasses import replace
from math import fsum

import pytest

from apps.knowledge_graph.retrieval import direct_seed_resolution
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectEntityMatchV1,
    DirectFailureReason,
    DirectResolutionTier,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    ready_generation_bundle_checksum,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

K = tuple(character * 64 for character in "123456789abcdef")


def _ready() -> ReadyGenerationBundleV1:
    generation = SelectedCollectionGenerationV1(
        K[0],
        K[1],
        K[2],
        K[3],
        K[4],
        "schema-v1",
        "projection-v1",
        "key-v1",
        1,
        K[5],
        "resolver-v1",
        K[6],
        K[7],
        "embed-v1",
    )
    documents = (AuthorizedProjectedDocumentV1(K[8], K[0], K[1]),)
    checksum = ready_generation_bundle_checksum((generation,), documents, K[9])
    return ReadyGenerationBundleV1((generation,), documents, K[9], checksum)


def _settings(*, embedding: bool = False, max_seeds: int = 32):
    return replace(
        load_hybrid_retrieval_settings({}),
        direct_embedding_enabled=embedding,
        direct_min_similarity=0.8,
        direct_winner_margin=0.05,
        graph_direct_max_seeds=max_seeds,
    )


# fmt: off
def _match(*, span: int, entity: str, component: str, tier: DirectResolutionTier, confidence: float = 1.0, similarity: float = 1.0) -> DirectEntityMatchV1:
    factor = {DirectResolutionTier.IDENTIFIER: 1.0, DirectResolutionTier.NAME: 0.95, DirectResolutionTier.ALIAS: 0.9, DirectResolutionTier.EMBEDDING: 0.8}[tier]
    return DirectEntityMatchV1(span, entity, component, "model", tier, confidence, similarity, confidence * factor * (similarity if tier is DirectResolutionTier.EMBEDDING else 1.0))
# fmt: on


class Repository:
    def __init__(self, tiers):
        self.tiers = tiers
        self.calls: list[tuple[str, int]] = []
        self.limits: list[int] = []

    def _get(self, name, span, limit):
        self.calls.append((name, span.start))
        self.limits.append(limit)
        return self.tiers.get((name, span.start), ())[:limit]

    def exact_identifier_matches(self, *, span, ready, limit):
        return self._get("identifier", span, limit)

    def canonical_name_matches(self, *, span, ready, limit):
        return self._get("name", span, limit)

    def indexed_alias_matches(self, *, span, ready, limit):
        return self._get("alias", span, limit)

    def embedding_matches(
        self,
        *,
        embedding,
        span,
        ontology_type,
        model_signature,
        ready,
        limit,
        minimum_similarity,
    ):
        return self._get("embedding", span, limit)

    def span_text(self, span):
        return f"model-{span.start}"


class TextRepository(Repository):
    def __init__(self, tiers, texts):
        super().__init__(tiers)
        self.texts = texts

    def span_text(self, span):
        return self.texts[span.start]


def test_deduplicates_normalized_surface_and_short_circuits_highest_confidence() -> (
    None
):
    lower = QueryEntitySpanV1("model", 0, 5, 0.4)
    higher = QueryEntitySpanV1("model", 10, 15, 0.9)
    zero = QueryEntitySpanV1("model", 20, 25, 0.0)
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
        {0: "Alpha", 10: "alpha", 20: "zero!"},
    )
    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=(lower, higher, zero),
        repository=repository,
        ready=_ready(),
        settings=_settings(),
        deadline=10.0,
    )
    assert outcome.failure_reason is None
    assert outcome.diagnostics.input_span_count == 3
    assert outcome.diagnostics.deduplicated_span_count == 2
    assert outcome.diagnostics.unresolved_span_count == 1
    assert outcome.matches[0].extraction_confidence == 0.9
    assert repository.calls == [("identifier", 10), ("name", 10)]


def test_automatic_component_ambiguity_and_same_component_best_member() -> None:
    spans = (
        QueryEntitySpanV1("model", 0, 5, 1.0),
        QueryEntitySpanV1("model", 6, 11, 1.0),
    )
    repository = Repository(
        {
            ("name", 0): (
                _match(
                    span=0, entity=K[0], component=K[4], tier=DirectResolutionTier.NAME
                ),
                _match(
                    span=0, entity=K[1], component=K[5], tier=DirectResolutionTier.NAME
                ),
            ),
            ("name", 6): (
                _match(
                    span=1, entity=K[3], component=K[6], tier=DirectResolutionTier.NAME
                ),
                _match(
                    span=1, entity=K[2], component=K[6], tier=DirectResolutionTier.NAME
                ),
            ),
        }
    )
    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=spans,
        repository=repository,
        ready=_ready(),
        settings=_settings(max_seeds=1),
        deadline=10.0,
    )
    assert outcome.ambiguities[0].component_count == 2
    assert outcome.matches[0].entity_key == K[2]
    assert outcome.matches[0].span_index == 1
    assert min(repository.limits) > 1


def test_uses_fsum_normalized_component_mass_and_opaque_tie_order() -> None:
    spans = tuple(
        QueryEntitySpanV1("model", index * 2, index * 2 + 1, 1.0) for index in range(3)
    )
    weights = (1.0, 1e-16, 1e-16)
    tiers = {
        ("identifier", span.start): (
            _match(
                span=index,
                entity=K[index],
                component=K[4 + index % 2],
                tier=DirectResolutionTier.IDENTIFIER,
                confidence=weights[index],
            ),
        )
        for index, span in enumerate(spans)
    }
    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=spans,
        repository=Repository(tiers),
        ready=_ready(),
        settings=_settings(),
        deadline=10.0,
    )
    total = fsum(weights)
    assert outcome.seeds[0].mass == fsum((weights[0], weights[2])) / total
    assert fsum(seed.mass for seed in outcome.seeds) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("similarities", "ambiguous", "resolved"),
    (((0.79,), 0, 0), ((0.90, 0.87), 1, 0), ((0.90, 0.84), 0, 1)),
)
def test_embedding_threshold_margin_and_transient_fallback(
    monkeypatch, similarities, ambiguous, resolved
) -> None:
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    matches = tuple(
        _match(
            span=0,
            entity=K[index],
            component=K[index + 3],
            tier=DirectResolutionTier.EMBEDDING,
            similarity=similarity,
        )
        for index, similarity in enumerate(similarities)
    )
    repository = Repository({("embedding", 0): matches})
    monkeypatch.setattr(
        direct_seed_resolution,
        "embed_unresolved_query_span",
        lambda **_kwargs: (0.0,) * 1024,
    )
    outcome = direct_seed_resolution.resolve_direct_seed_components(
        spans=(span,),
        repository=repository,
        ready=_ready(),
        settings=_settings(embedding=True),
        deadline=10.0,
    )
    assert outcome.diagnostics.ambiguous_span_count == ambiguous
    assert outcome.diagnostics.resolved_span_count == resolved
    assert outcome.failure_reason is (
        None if resolved else DirectFailureReason.DIRECT_NO_SEEDS
    )


def test_embedding_failure_preserves_exact_matches_and_disables_only_fallback(
    monkeypatch,
) -> None:
    spans = tuple(
        QueryEntitySpanV1("model", start, start + 5, 1.0) for start in (0, 6, 12)
    )
    repository = TextRepository(
        {
            ("name", start): (
                _match(
                    span=index,
                    entity=K[index],
                    component=K[index + 3],
                    tier=DirectResolutionTier.NAME,
                ),
            )
            for index, start in ((0, 0), (2, 12))
        },
        {0: "alpha", 6: "beta", 12: "gamma"},
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
    assert ("name", 12) in repository.calls


# fmt: off
def test_applies_configured_seed_cap_globally_after_resolution() -> None:
    spans = tuple(QueryEntitySpanV1("model", index * 2, index * 2 + 1, 1.0) for index in range(65))
    tiers = {("identifier", span.start): (_match(span=index, entity=f"{index + 1:064x}", component=f"{index + 1:064x}", tier=DirectResolutionTier.IDENTIFIER),) for index, span in enumerate(spans)}
    outcome = direct_seed_resolution.resolve_direct_seed_components(spans=spans, repository=Repository(tiers), ready=_ready(), settings=_settings(max_seeds=3), deadline=10.0)
    assert tuple(seed.component_key for seed in outcome.seeds) == tuple(f"{index:064x}" for index in range(1, 4))
    assert len(outcome.matches) == outcome.diagnostics.resolved_span_count == 3
    assert outcome.diagnostics.unresolved_span_count == 62
