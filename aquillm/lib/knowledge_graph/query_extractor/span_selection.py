"""Select bounded, canonical non-overlapping entity spans."""

from __future__ import annotations

from .contracts import QueryEntitySpanV1


def _canonical_spans(entities: object, maximum: int) -> tuple[QueryEntitySpanV1, ...]:
    candidates: dict[tuple[int, int, str], QueryEntitySpanV1] = {}
    for entity in entities:  # type: ignore[union-attr]
        span = QueryEntitySpanV1(
            entity.entity_type,
            entity.start,
            entity.end,
            float(entity.confidence),
        )
        key = (span.start, span.end, span.ontology_type)
        previous = candidates.get(key)
        if previous is None or span.confidence > previous.confidence:
            candidates[key] = span
    selected: list[QueryEntitySpanV1] = []
    for span in sorted(
        candidates.values(), key=lambda row: (row.start, row.end, row.ontology_type)
    ):
        if selected and selected[-1].end > span.start:
            continue
        selected.append(span)
        if len(selected) == maximum:
            break
    return tuple(selected)
