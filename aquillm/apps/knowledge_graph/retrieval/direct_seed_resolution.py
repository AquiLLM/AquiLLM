"""Deterministic tiered resolution of query spans to direct graph seeds."""

from __future__ import annotations

from dataclasses import replace
from math import fsum

from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectEntityMatchV1,
    DirectFailureReason,
    DirectResolutionTier,
    DirectSeedAmbiguityV1,
    DirectSeedDiagnosticsV1,
    DirectSeedOutcomeV1,
    ResolvedDirectSeedV1,
)
from apps.knowledge_graph.retrieval.query_embedding import embed_unresolved_query_span
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1


def _deduplicate(spans: tuple[QueryEntitySpanV1, ...]) -> tuple[QueryEntitySpanV1, ...]:
    seen: set[tuple[int, int, str]] = set()
    rows = []
    for span in spans:
        key = span.start, span.end, span.ontology_type
        if key not in seen:
            seen.add(key)
            rows.append(span)
    return tuple(rows)


def _best_exact(
    rows: tuple[DirectEntityMatchV1, ...], span_index: int
) -> tuple[DirectEntityMatchV1 | None, DirectSeedAmbiguityV1 | None]:
    components = {row.component_key for row in rows}
    if len(components) > 1:
        return None, DirectSeedAmbiguityV1(
            span_index, rows[0].tier, len(components), len(rows)
        )
    best = min(rows, key=lambda row: (-row.match_weight, row.entity_key))
    return replace(best, span_index=span_index), None


def _best_embedding(
    rows: tuple[DirectEntityMatchV1, ...],
    *,
    span_index: int,
    minimum: float,
    margin: float,
) -> tuple[DirectEntityMatchV1 | None, DirectSeedAmbiguityV1 | None]:
    eligible = tuple(row for row in rows if row.similarity >= minimum)
    if not eligible:
        return None, None
    best_by_component: dict[str, DirectEntityMatchV1] = {}
    for row in eligible:
        current = best_by_component.get(row.component_key)
        if current is None or (-row.similarity, row.entity_key) < (
            -current.similarity,
            current.entity_key,
        ):
            best_by_component[row.component_key] = row
    ordered = sorted(
        best_by_component.values(), key=lambda row: (-row.similarity, row.entity_key)
    )
    if len(ordered) > 1 and ordered[0].similarity - ordered[1].similarity < margin:
        return None, DirectSeedAmbiguityV1(
            span_index, DirectResolutionTier.EMBEDDING, len(ordered), len(eligible)
        )
    return replace(ordered[0], span_index=span_index), None


def _diagnostics(
    *,
    total: int,
    deduplicated: int,
    matches: int,
    ambiguities: int,
    embedding_attempts: int,
    embedding_matches: int,
) -> DirectSeedDiagnosticsV1:
    return DirectSeedDiagnosticsV1(
        input_span_count=total,
        deduplicated_span_count=deduplicated,
        resolved_span_count=matches,
        ambiguous_span_count=ambiguities,
        unresolved_span_count=deduplicated - matches - ambiguities,
        embedding_attempt_count=embedding_attempts,
        embedding_match_count=embedding_matches,
    )


def _failure(
    reason: DirectFailureReason,
    *,
    total: int,
    deduplicated: int,
    embedding_attempts: int = 0,
) -> DirectSeedOutcomeV1:
    diagnostics = _diagnostics(
        total=total,
        deduplicated=deduplicated,
        matches=0,
        ambiguities=0,
        embedding_attempts=embedding_attempts,
        embedding_matches=0,
    )
    return DirectSeedOutcomeV1((), (), (), diagnostics, reason)


def resolve_direct_seed_components(
    *,
    spans: tuple[QueryEntitySpanV1, ...],
    repository,
    ready,
    settings,
    deadline: float,
) -> DirectSeedOutcomeV1:
    if type(spans) is not tuple or any(
        type(row) is not QueryEntitySpanV1 for row in spans
    ):
        raise TypeError("spans must contain exact QueryEntitySpanV1 values")
    if len(spans) > 128:
        raise ValueError("spans exceed the hard cap")
    deduplicated = _deduplicate(spans)
    matches: list[DirectEntityMatchV1] = []
    ambiguities: list[DirectSeedAmbiguityV1] = []
    embedding_attempts = 0
    embedding_matches = 0
    exact_tiers = (
        (DirectResolutionTier.IDENTIFIER, repository.exact_identifier_matches),
        (DirectResolutionTier.NAME, repository.canonical_name_matches),
        (DirectResolutionTier.ALIAS, repository.indexed_alias_matches),
    )
    limit = settings.graph_direct_max_seeds
    for span_index, span in enumerate(deduplicated):
        selected: DirectEntityMatchV1 | None = None
        ambiguity: DirectSeedAmbiguityV1 | None = None
        for _tier, lookup in exact_tiers:
            rows = lookup(span=span, ready=ready, limit=limit)
            if rows:
                selected, ambiguity = _best_exact(rows, span_index)
                break
        if selected is None and ambiguity is None and settings.direct_embedding_enabled:
            embedding_attempts += 1
            try:
                signature = ready.selected_generations[0].embedding_model_signature
                if any(
                    row.embedding_model_signature != signature
                    for row in ready.selected_generations
                ):
                    raise RuntimeError("mixed embedding signatures")
                embedding = embed_unresolved_query_span(
                    text=repository.span_text(span),
                    expected_signature=signature,
                    deadline=deadline,
                )
                rows = repository.embedding_matches(
                    embedding=embedding,
                    span=span,
                    ontology_type=span.ontology_type,
                    model_signature=signature,
                    ready=ready,
                    limit=limit,
                )
            except (RuntimeError, TimeoutError, TypeError, ValueError):
                return _failure(
                    DirectFailureReason.DIRECT_EMBEDDING_UNAVAILABLE,
                    total=len(spans),
                    deduplicated=len(deduplicated),
                    embedding_attempts=embedding_attempts,
                )
            selected, ambiguity = _best_embedding(
                rows,
                span_index=span_index,
                minimum=settings.direct_min_similarity,
                margin=settings.direct_winner_margin,
            )
            if selected is not None or ambiguity is not None:
                embedding_matches += 1
        if selected is not None:
            matches.append(selected)
        elif ambiguity is not None:
            ambiguities.append(ambiguity)
    matches.sort(
        key=lambda row: (
            row.span_index,
            row.tier.priority,
            row.component_key,
            row.entity_key,
        )
    )
    ambiguities.sort(key=lambda row: row.span_index)
    diagnostics = _diagnostics(
        total=len(spans),
        deduplicated=len(deduplicated),
        matches=len(matches),
        ambiguities=len(ambiguities),
        embedding_attempts=embedding_attempts,
        embedding_matches=embedding_matches,
    )
    if not matches:
        return DirectSeedOutcomeV1(
            (),
            (),
            tuple(ambiguities),
            diagnostics,
            DirectFailureReason.DIRECT_NO_SEEDS,
        )
    total_weight = fsum(row.match_weight for row in matches)
    members: dict[str, set[str]] = {}
    for row in matches:
        members.setdefault(row.component_key, set()).add(row.entity_key)
    seeds = [
        ResolvedDirectSeedV1(
            component,
            tuple(sorted(entity_keys)),
            fsum(row.match_weight for row in matches if row.component_key == component)
            / total_weight,
        )
        for component, entity_keys in members.items()
    ]
    seeds.sort(key=lambda row: (-row.mass, row.member_entity_keys[0]))
    return DirectSeedOutcomeV1(
        tuple(matches), tuple(seeds), tuple(ambiguities), diagnostics, None
    )


__all__ = ["resolve_direct_seed_components"]
