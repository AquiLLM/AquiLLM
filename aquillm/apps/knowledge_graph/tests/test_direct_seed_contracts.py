# ruff: noqa: E501
# fmt: off
import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from math import fsum

import pytest

from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectEntityMatchV1,
    DirectFailureReason,
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
    DirectSeedAmbiguityV1,
    DirectSeedDiagnosticsV1,
    DirectSeedOutcomeV1,
    ResolvedDirectSeedV1,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

K = tuple(character * 64 for character in "123456789abcdef")
def _match(
    span_index: int = 0,
    component_key: str = K[1],
) -> DirectEntityMatchV1:
    return DirectEntityMatchV1(
        span_index=span_index,
        entity_key=K[0],
        component_key=component_key,
        ontology_type="person",
        tier=DirectResolutionTier.NAME,
        extraction_confidence=0.8,
        similarity=1.0,
        match_weight=0.76,
    )
def _diagnostics(**changes: int) -> DirectSeedDiagnosticsV1:
    values = {
        "input_span_count": 1,
        "deduplicated_span_count": 1,
        "resolved_span_count": 1,
        "ambiguous_span_count": 0,
        "unresolved_span_count": 0,
        "embedding_attempt_count": 0,
        "embedding_match_count": 0,
    }
    values.update(changes)
    return DirectSeedDiagnosticsV1(**values)
def _outcome() -> DirectSeedOutcomeV1:
    return DirectSeedOutcomeV1(
        matches=(_match(),),
        seeds=(ResolvedDirectSeedV1(K[1], (K[0],), 1.0),),
        ambiguities=(),
        diagnostics=_diagnostics(),
        failure_reason=None,
    )
def test_resolution_tiers_priority_fields_and_exact_failure_values() -> None:
    assert tuple(DirectResolutionTier) == (
        "identifier",
        "name",
        "alias",
        "embedding",
    )
    assert tuple(tier.priority for tier in DirectResolutionTier) == (0, 1, 2, 3)
    assert tuple(DirectFailureReason) == (
        "extractor_timeout",
        "extractor_auth",
        "extractor_provenance",
        "mixed_ontology",
        "direct_seed_invalid",
        "direct_no_seeds",
        "direct_embedding_unavailable",
        "direct_topology_timeout",
        "direct_topology_invalid",
        "direct_ppr_invalid",
    )
    assert tuple(field.name for field in fields(DirectEntityMatchV1)) == (
        "span_index",
        "entity_key",
        "component_key",
        "ontology_type",
        "tier",
        "extraction_confidence",
        "similarity",
        "match_weight",
    )
    assert tuple(field.name for field in fields(DirectResolutionSpanInputV1)) == ("span", "text")

def test_matches_pin_tier_weights_exact_types_and_no_query_or_database_ids() -> None:
    expected = {
        DirectResolutionTier.IDENTIFIER: 0.8,
        DirectResolutionTier.NAME: 0.76,
        DirectResolutionTier.ALIAS: 0.72,
        DirectResolutionTier.EMBEDDING: 0.512,
    }
    for tier, weight in expected.items():
        similarity = 0.8 if tier is DirectResolutionTier.EMBEDDING else 1.0
        match = replace(
            _match(),
            tier=tier,
            similarity=similarity,
            match_weight=weight,
        )
        assert match.match_weight == weight
    assert not {
        "query",
        "text",
        "span_text",
        "entity_id",
        "document_id",
        "chunk_id",
    } & {field.name for field in fields(DirectEntityMatchV1)}
    with pytest.raises(TypeError):
        replace(_match(), span_index=True)
    with pytest.raises(TypeError):
        replace(_match(), similarity=1)
    with pytest.raises(ValueError):
        replace(_match(), tier=DirectResolutionTier.EMBEDDING)
    with pytest.raises((TypeError, ValueError)):
        replace(_match(), entity_key=K[10].upper())
def test_seed_mass_member_order_best_match_and_diagnostic_coherence() -> None:
    outcome = _outcome()
    assert fsum(seed.mass for seed in outcome.seeds) == 1.0
    with pytest.raises(FrozenInstanceError):
        outcome.seeds = ()  # type: ignore[misc]
    assert not hasattr(outcome, "__dict__")
    with pytest.raises(ValueError, match="sorted"):
        ResolvedDirectSeedV1(K[2], (K[2], K[0]), 1.0)
    with pytest.raises(ValueError, match="mass"):
        replace(outcome, seeds=(replace(outcome.seeds[0], mass=0.9),))
    with pytest.raises(ValueError, match="unique|span"):
        replace(outcome, matches=(outcome.matches[0], outcome.matches[0]))
    with pytest.raises(ValueError, match="diagnostic"):
        replace(outcome, diagnostics=_diagnostics(resolved_span_count=0))


def test_local_resolution_span_is_redacted_and_not_serializable() -> None:
    class Text(str):
        pass

    span = QueryEntitySpanV1("person", 0, 6, 0.9)
    local = DirectResolutionSpanInputV1(span, "Élodie")
    assert local.text == "Élodie"
    assert "Élodie" not in repr(local) and "Élodie" not in str(local)
    with pytest.raises(TypeError):
        pickle.dumps(local)
    with pytest.raises(TypeError):
        json.dumps(local)
    with pytest.raises(ValueError, match="length"):
        replace(local, text="Élodi")
    with pytest.raises(TypeError):
        replace(local, text=Text("Élodie"))
    for text in ("a\nbcde", "a\x7fbcde"):
        with pytest.raises(ValueError, match="control"):
            replace(local, text=text)
    with pytest.raises(ValueError, match="UTF-8"):
        DirectResolutionSpanInputV1(QueryEntitySpanV1("person", 0, 1, 0.9), chr(0xD800))
    oversized = "😀" * 8193
    with pytest.raises(ValueError, match="bound"):
        DirectResolutionSpanInputV1(QueryEntitySpanV1("person", 0, len(oversized), 0.9), oversized)


def test_component_mass_is_the_normalized_fsum_of_best_match_weights() -> None:
    first = _match()
    second = replace(
        _match(span_index=1, component_key=K[3]),
        entity_key=K[2],
    )
    outcome = DirectSeedOutcomeV1(
        matches=(first, second),
        seeds=(
            ResolvedDirectSeedV1(K[1], (K[0],), 0.5),
            ResolvedDirectSeedV1(K[3], (K[2],), 0.5),
        ),
        ambiguities=(),
        diagnostics=_diagnostics(
            input_span_count=2,
            deduplicated_span_count=2,
            resolved_span_count=2,
        ),
        failure_reason=None,
    )
    with pytest.raises(ValueError, match="normalized|mass"):
        replace(
            outcome,
            seeds=(
                replace(outcome.seeds[0], mass=0.6),
                replace(outcome.seeds[1], mass=0.4),
            ),
        )
    with pytest.raises(ValueError, match="unique|partition"):
        replace(
            outcome,
            seeds=(
                replace(outcome.seeds[0], member_entity_keys=(K[0], K[4])),
                replace(outcome.seeds[1], member_entity_keys=(K[2], K[4])),
            ),)


def test_seed_rows_sort_by_descending_mass_then_smallest_member_key() -> None:
    low = replace(_match(), extraction_confidence=0.4, match_weight=0.38)
    high = replace(
        _match(span_index=1, component_key=K[3]),
        entity_key=K[2],
    )
    total = fsum((low.match_weight, high.match_weight))
    seeds = (
        ResolvedDirectSeedV1(K[3], (K[2],), high.match_weight / total),
        ResolvedDirectSeedV1(K[1], (K[0],), low.match_weight / total),
    )
    outcome = DirectSeedOutcomeV1(
        (low, high),
        seeds,
        (),
        _diagnostics(
            input_span_count=2,
            deduplicated_span_count=2,
            resolved_span_count=2,
        ),
        None,
    )
    with pytest.raises(ValueError, match="sorted"):
        replace(outcome, seeds=tuple(reversed(seeds)))


def test_ambiguities_are_bounded_safe_disjoint_and_failure_is_closed() -> None:
    ambiguity = DirectSeedAmbiguityV1(
        span_index=0,
        tier=DirectResolutionTier.IDENTIFIER,
        component_count=2,
        candidate_count=3,
    )
    failed = DirectSeedOutcomeV1(
        matches=(),
        seeds=(),
        ambiguities=(ambiguity,),
        diagnostics=_diagnostics(
            resolved_span_count=0,
            ambiguous_span_count=1,
        ),
        failure_reason=DirectFailureReason.DIRECT_NO_SEEDS,
    )
    assert failed.failure_reason is DirectFailureReason.DIRECT_NO_SEEDS
    assert not any(
        name.endswith("_id") or "text" in name
        for name in (field.name for field in fields(DirectSeedDiagnosticsV1))
    )
    with pytest.raises(TypeError):
        replace(failed, failure_reason="direct_no_seeds")
    with pytest.raises(ValueError, match="disjoint"):
        replace(_outcome(), ambiguities=(ambiguity,))
    with pytest.raises(ValueError, match="failure"):
        replace(_outcome(), failure_reason=DirectFailureReason.DIRECT_SEED_INVALID)
    second_tier = replace(ambiguity, tier=DirectResolutionTier.NAME)
    with pytest.raises(ValueError, match="span|ambigu"):
        DirectSeedOutcomeV1(
            (),
            (),
            (ambiguity, second_tier),
            _diagnostics(
                input_span_count=2,
                deduplicated_span_count=2,
                resolved_span_count=0,
                ambiguous_span_count=2,
            ),
            DirectFailureReason.DIRECT_NO_SEEDS,
        )
    for changes in (
        {"embedding_attempt_count": 2},
        {
            "input_span_count": 0,
            "deduplicated_span_count": 0,
            "resolved_span_count": 0,
            "embedding_attempt_count": 1,
        },
    ):
        with pytest.raises(ValueError, match="embedding"):
            _diagnostics(**changes)


def test_embedding_diagnostics_bind_authoritative_tier_outcomes() -> None:
    with pytest.raises(ValueError, match="embedding"):
        replace(_outcome(), diagnostics=_diagnostics(embedding_attempt_count=1, embedding_match_count=1))
    embedding_match = replace(_match(), tier=DirectResolutionTier.EMBEDDING, similarity=0.8, match_weight=0.512)
    DirectSeedOutcomeV1(
        matches=(embedding_match,),
        seeds=(ResolvedDirectSeedV1(K[1], (K[0],), 1.0),),
        ambiguities=(),
        diagnostics=_diagnostics(embedding_attempt_count=1, embedding_match_count=1),
        failure_reason=None,
    )
    embedding_ambiguity = DirectSeedAmbiguityV1(0, DirectResolutionTier.EMBEDDING, 2, 2)
    DirectSeedOutcomeV1(
        matches=(),
        seeds=(),
        ambiguities=(embedding_ambiguity,),
        diagnostics=_diagnostics(resolved_span_count=0, ambiguous_span_count=1, embedding_attempt_count=1, embedding_match_count=1),
        failure_reason=DirectFailureReason.DIRECT_NO_SEEDS,
    )
