"""Deterministic graph-seed construction from baseline retrieval ranks."""

from __future__ import annotations

from collections.abc import Sequence
from math import fsum, isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.knowledge_graph.retrieval import (
        GraphExpansionConfig,
        GraphExpansionSeed,
    )

_DATABASE_ID_MAX = 2**63 - 1


def _positive_chunk_id(candidate: object) -> int:
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is not int or not 1 <= identifier <= _DATABASE_ID_MAX:
        raise ValueError("candidate chunk IDs must be positive database integers")
    return identifier


def _source_first_ranks(rows: Sequence[object]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for rank, row in enumerate(rows, start=1):
        identifier = _positive_chunk_id(row)
        ranks.setdefault(identifier, rank)
    return ranks


def build_graph_seeds(
    vector_rows: Sequence[object],
    trigram_rows: Sequence[object],
    exact_rows: Sequence[object],
    graph_config: GraphExpansionConfig,
) -> tuple[GraphExpansionSeed, ...]:
    """Fuse source ranks and normalize a bounded restart distribution."""

    from apps.knowledge_graph.retrieval import (
        GraphExpansionConfig,
        GraphExpansionSeed,
    )

    if type(graph_config) is not GraphExpansionConfig:
        raise ValueError("graph_config must be an exact GraphExpansionConfig")
    contributions: dict[int, list[float]] = {}
    for rows in (vector_rows, trigram_rows, exact_rows):
        for identifier, rank in _source_first_ranks(rows).items():
            contribution = 1.0 / (graph_config.rrf_k + rank)
            if not isfinite(contribution) or contribution <= 0.0:
                raise ValueError("RRF produced a non-finite seed contribution")
            contributions.setdefault(identifier, []).append(contribution)

    weighted = [
        (identifier, fsum(values)) for identifier, values in contributions.items()
    ]
    if any(not isfinite(weight) or weight <= 0.0 for _identifier, weight in weighted):
        raise ValueError("RRF produced a non-finite seed weight")
    weighted.sort(key=lambda item: (-item[1], item[0]))
    selected = weighted[: graph_config.max_seeds]
    if not selected:
        return ()
    total = fsum(weight for _identifier, weight in selected)
    if not isfinite(total) or total <= 0.0:
        raise ValueError("RRF produced a non-finite normalization total")
    return tuple(
        GraphExpansionSeed(
            chunk_id=identifier,
            rank=rank,
            restart_weight=weight / total,
        )
        for rank, (identifier, weight) in enumerate(selected, start=1)
    )


__all__ = ["build_graph_seeds"]
