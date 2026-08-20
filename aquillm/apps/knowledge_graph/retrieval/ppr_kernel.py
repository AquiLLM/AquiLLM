"""Generic deterministic PageRank recurrence for provider-neutral node keys."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any, Protocol


class PPRKernelConfig(Protocol):
    """Structural subset of an algorithm config consumed by the kernel."""

    max_nodes: int
    max_edges: int
    ppr_restart: float
    ppr_iterations: int


@dataclass(frozen=True, slots=True)
class WeightedEdge[NodeKeyT: Hashable]:
    source: NodeKeyT
    target: NodeKeyT
    weight: float


@dataclass(frozen=True, slots=True)
class PPRKernelResult[NodeKeyT: Hashable]:
    scores: tuple[tuple[NodeKeyT, float], ...]
    transition_rows: tuple[tuple[NodeKeyT, tuple[tuple[NodeKeyT, float], ...]], ...]

    def score_map(self) -> dict[NodeKeyT, float]:
        return dict(self.scores)


def safe_fsum(values, label: str) -> float:
    """Sum validated nonnegative terms with deterministic finite failure."""

    try:
        total = fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} produced non-finite math") from error
    if not isfinite(total) or total < 0.0:
        raise ValueError(f"{label} produced non-finite math")
    return total


def run_normalized_ppr_kernel[NodeKeyT: Hashable](
    *,
    ordered_nodes: tuple[NodeKeyT, ...],
    transition_rows: Mapping[NodeKeyT, tuple[tuple[NodeKeyT, float], ...]],
    restart: Mapping[NodeKeyT, float],
    restart_probability: float,
    iterations: int,
    deadline_check: Callable[[], None] | None = None,
) -> dict[NodeKeyT, float]:
    """Run the fixed recurrence over prevalidated, normalized inputs."""

    scores = dict(restart)
    damping = 1.0 - restart_probability
    for _ in range(iterations):
        if deadline_check is not None:
            deadline_check()
        incoming: dict[NodeKeyT, list[float]] = defaultdict(list)
        dangling_scores: list[float] = []
        for source in ordered_nodes:
            row = transition_rows[source]
            if not row:
                dangling_scores.append(scores[source])
                continue
            for target, share in row:
                incoming[target].append(scores[source] * share)
        dangling_mass = safe_fsum(dangling_scores, "dangling mass")
        next_scores: dict[NodeKeyT, float] = {}
        for target in ordered_nodes:
            propagated = safe_fsum(incoming[target], "incoming transition mass")
            restart_mass = restart[target]
            score = restart_probability * restart_mass + damping * (
                propagated + dangling_mass * restart_mass
            )
            if not isfinite(score) or score < 0.0:
                raise ValueError("PPR iteration produced non-finite math")
            next_scores[target] = score
        scores = next_scores
        if deadline_check is not None:
            deadline_check()
    return scores


def _weight(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def run_ppr_kernel[NodeKeyT: Hashable](
    *,
    nodes: tuple[NodeKeyT, ...],
    edges: tuple[WeightedEdge[NodeKeyT], ...],
    seeds: Mapping[NodeKeyT, float],
    config: PPRKernelConfig,
    order_key: Callable[[NodeKeyT], Any],
    deadline_check: Callable[[], None] | None = None,
) -> PPRKernelResult[NodeKeyT]:
    """Validate generic keys, normalize in caller-defined order, and rank."""

    if type(nodes) is not tuple or len(nodes) > config.max_nodes:
        raise ValueError("nodes must be an exact tuple within max_nodes")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    ordered_nodes = tuple(sorted(nodes, key=order_key))
    if type(edges) is not tuple or len(edges) > config.max_edges:
        raise ValueError("edges must be an exact tuple within max_edges")
    node_set = set(ordered_nodes)
    grouped: dict[NodeKeyT, dict[NodeKeyT, list[float]]] = {
        node: {} for node in ordered_nodes
    }
    for edge in edges:
        if type(edge) is not WeightedEdge:
            raise TypeError("edges must contain exact WeightedEdge rows")
        if edge.source not in node_set or edge.target not in node_set:
            raise ValueError("edges must reference nodes")
        weight = _weight(edge.weight, "edge weight")
        if weight:
            grouped[edge.source].setdefault(edge.target, []).append(weight)
    rows: dict[NodeKeyT, tuple[tuple[NodeKeyT, float], ...]] = {}
    for source in ordered_nodes:
        targets = sorted(grouped[source], key=order_key)
        combined = {
            target: safe_fsum(sorted(grouped[source][target]), "edge sum")
            for target in targets
        }
        total = safe_fsum(combined.values(), "transition row sum")
        rows[source] = (
            tuple((target, combined[target] / total) for target in targets)
            if total
            else ()
        )
    if not isinstance(seeds, Mapping) or not seeds:
        raise ValueError("seeds must be a nonempty mapping")
    if any(node not in node_set for node in seeds):
        raise ValueError("seeds must reference nodes")
    seed_weights = {
        node: _weight(seeds.get(node, 0.0), "seed weight") for node in ordered_nodes
    }
    total = safe_fsum(seed_weights.values(), "seed weight sum")
    if not total:
        raise ValueError("seeds must have a positive sum")
    restart = {node: seed_weights[node] / total for node in ordered_nodes}
    scores = run_normalized_ppr_kernel(
        ordered_nodes=ordered_nodes,
        transition_rows=rows,
        restart=restart,
        restart_probability=config.ppr_restart,
        iterations=config.ppr_iterations,
        deadline_check=deadline_check,
    )
    return PPRKernelResult(
        scores=tuple(scores.items()), transition_rows=tuple(rows.items())
    )


__all__ = [
    "PPRKernelConfig",
    "PPRKernelResult",
    "WeightedEdge",
    "run_ppr_kernel",
]
