# ruff: noqa: E501,I001
# fmt: off
import copy
import json
import pickle
from dataclasses import FrozenInstanceError, asdict, astuple, fields, is_dataclass, replace
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
def _match(span_index: int = 0, component_key: str = K[1]) -> DirectEntityMatchV1:
    return DirectEntityMatchV1(span_index, K[0], component_key, "person", DirectResolutionTier.NAME, 0.8, 1.0, 0.76)
def _diagnostics(**changes: int) -> DirectSeedDiagnosticsV1:
    values = {"input_span_count": 1, "deduplicated_span_count": 1, "resolved_span_count": 1, "ambiguous_span_count": 0, "unresolved_span_count": 0, "embedding_attempt_count": 0, "embedding_match_count": 0}
    values.update(changes)
    return DirectSeedDiagnosticsV1(**values)
def _outcome() -> DirectSeedOutcomeV1:
    return DirectSeedOutcomeV1((_match(),), (ResolvedDirectSeedV1(K[1], (K[0],), 1.0),), (), _diagnostics(), None)
def _failure(reason, diagnostics, ambiguities=()) -> DirectSeedOutcomeV1:
    return DirectSeedOutcomeV1((), (), ambiguities, diagnostics, reason)
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
    assert not is_dataclass(DirectResolutionSpanInputV1)
    assert DirectResolutionSpanInputV1.__slots__ == ("_span", "_text")
def test_matches_pin_tier_weights_exact_types_and_no_query_or_database_ids() -> None:
    expected = {
        DirectResolutionTier.IDENTIFIER: 0.8,
        DirectResolutionTier.NAME: 0.8 * 0.95,
        DirectResolutionTier.ALIAS: 0.8 * 0.9,
        DirectResolutionTier.EMBEDDING: 0.8 * 0.8 * 0.8,
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
    for changes in ({"extraction_confidence": 0.0, "match_weight": 1e-13}, {"tier": DirectResolutionTier.EMBEDDING, "similarity": 0.0, "match_weight": 1e-13}, {"match_weight": _match().match_weight + 5e-13}):
        with pytest.raises(ValueError, match="positive|weight"):
            replace(_match(), **changes)
    with pytest.raises((TypeError, ValueError)):
        replace(_match(), entity_key=K[10].upper())
    with pytest.raises(ValueError, match="UTF-8"):
        replace(_match(), ontology_type=chr(0xD800))
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
    with pytest.raises(ValueError, match="span"):
        replace(outcome, matches=(replace(outcome.matches[0], span_index=1),))
def test_local_resolution_span_is_redacted_and_not_serializable() -> None:
    class Text(str):
        pass
    span = QueryEntitySpanV1("person", 0, 6, 0.9)
    local = DirectResolutionSpanInputV1(span, "Élodie")
    assert local.text == "Élodie"
    assert "Élodie" not in repr(local) and "Élodie" not in str(local)
    for operation in (lambda: pickle.dumps(local), lambda: copy.copy(local), lambda: copy.deepcopy(local), lambda: asdict(local), lambda: astuple(local), lambda: json.dumps(local)):
        with pytest.raises(TypeError):
            operation()
    for operation in (lambda: setattr(local, "text", "secret"), lambda: setattr(local, "_text", "secret"), lambda: delattr(local, "_text")):
        with pytest.raises(AttributeError):
            operation()
    with pytest.raises(ValueError, match="length"):
        DirectResolutionSpanInputV1(span, "Élodi")
    with pytest.raises(TypeError):
        DirectResolutionSpanInputV1(span, Text("Élodie"))
    for text in ("a\nbcde", "a\x7fbcde"):
        with pytest.raises(ValueError, match="control"):
            DirectResolutionSpanInputV1(span, text)
    with pytest.raises(ValueError, match="UTF-8"):
        DirectResolutionSpanInputV1(QueryEntitySpanV1("person", 0, 1, 0.9), chr(0xD800))
    oversized = "😀" * 4097
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
    with pytest.raises(ValueError, match="span"):
        replace(failed, ambiguities=(replace(ambiguity, span_index=1),))
    for reason in (DirectFailureReason.DIRECT_TOPOLOGY_TIMEOUT, DirectFailureReason.DIRECT_TOPOLOGY_INVALID, DirectFailureReason.DIRECT_PPR_INVALID):
        with pytest.raises(ValueError, match="stage|failure"):
            replace(failed, failure_reason=reason)
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
    embedding_match = replace(_match(), tier=DirectResolutionTier.EMBEDDING, similarity=0.8, match_weight=0.8 * 0.8 * 0.8)
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
def test_failure_reasons_bind_lifecycle_diagnostics() -> None:
    zero = _diagnostics(input_span_count=0, deduplicated_span_count=0, resolved_span_count=0)
    nonzero = _diagnostics(input_span_count=1, deduplicated_span_count=0, resolved_span_count=0)
    for reason in (DirectFailureReason.EXTRACTOR_TIMEOUT, DirectFailureReason.EXTRACTOR_AUTH, DirectFailureReason.EXTRACTOR_PROVENANCE):
        _failure(reason, zero)
        pytest.raises(ValueError, _failure, reason, nonzero)
    unresolved = _diagnostics(resolved_span_count=0, unresolved_span_count=1)
    ambiguity = DirectSeedAmbiguityV1(0, DirectResolutionTier.NAME, 2, 2)
    ambiguous = _diagnostics(resolved_span_count=0, ambiguous_span_count=1)
    pytest.raises(ValueError, _failure, DirectFailureReason.EXTRACTOR_AUTH, ambiguous, (ambiguity,))
    for reason in (DirectFailureReason.MIXED_ONTOLOGY, DirectFailureReason.DIRECT_SEED_INVALID):
        _failure(reason, unresolved)
        pytest.raises(ValueError, _failure, reason, replace(unresolved, embedding_attempt_count=1))
        pytest.raises(ValueError, _failure, reason, ambiguous, (ambiguity,))
    _failure(DirectFailureReason.DIRECT_NO_SEEDS, unresolved)
    _failure(DirectFailureReason.DIRECT_NO_SEEDS, ambiguous, (ambiguity,))
    embedding_diag = replace(ambiguous, embedding_attempt_count=1)
    _failure(DirectFailureReason.DIRECT_EMBEDDING_UNAVAILABLE, embedding_diag, (ambiguity,))
    pytest.raises(ValueError, _failure, DirectFailureReason.DIRECT_EMBEDDING_UNAVAILABLE, ambiguous, (ambiguity,))
    embedding_ambiguity = replace(ambiguity, tier=DirectResolutionTier.EMBEDDING)
    pytest.raises(ValueError, _failure, DirectFailureReason.DIRECT_EMBEDDING_UNAVAILABLE, replace(embedding_diag, embedding_match_count=1), (embedding_ambiguity,))
