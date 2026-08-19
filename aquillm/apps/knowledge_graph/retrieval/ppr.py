"""Deterministic, ORM-free numerical primitives for ``ppr_v1`` retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import fsum, isfinite, log1p
from time import monotonic
from typing import Literal, cast

ALGORITHM_VERSION = "ppr_v1"
TRANSITION_VERSION = "ppr_transition_v1"
EVIDENCE_VERSION = "ppr_evidence_v1"
SEED_VERSION = "rrf_seed_v1"

REVERSE_DIRECTION_FACTOR = 0.35
SUPPORT_CAP = 32
UTILITY_FLOOR = 0.5
MENTION_FACTOR = 0.25

MAX_PPR_NODES = 200
MAX_PPR_EDGES = 1_000
MAX_PPR_ITERATIONS = 8

type StableNodeKey = (
    tuple[Literal["canonical"], int] | tuple[Literal["local"], str]
)
type WeightedTransitionRows = Mapping[
    StableNodeKey, Iterable[tuple[StableNodeKey, int | float]]
]
type NormalizedTransitionRows = dict[
    StableNodeKey, tuple[tuple[StableNodeKey, float], ...]
]


@dataclass(frozen=True, slots=True, repr=False)
class _MonotonicDeadline:
    """Package-private shared deadline with an injectable monotonic test clock."""

    expires_at: float
    clock: Callable[[], float]

    def __post_init__(self) -> None:
        if type(self.expires_at) not in (int, float):
            raise ValueError("deadline must be a finite monotonic timestamp")
        expires_at = float(self.expires_at)
        if not isfinite(expires_at):
            raise ValueError("deadline must be a finite monotonic timestamp")
        if not callable(self.clock):
            raise ValueError("deadline clock must be callable")
        object.__setattr__(self, "expires_at", expires_at)

    @classmethod
    def after_ms(
        cls,
        timeout_ms: int,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> _MonotonicDeadline:
        if type(timeout_ms) is not int or timeout_ms < 1:
            raise ValueError("timeout_ms must be a positive exact integer")
        started_at = clock()
        if type(started_at) not in (int, float) or not isfinite(float(started_at)):
            raise ValueError("deadline clock must return a finite number")
        return cls(float(started_at) + timeout_ms / 1_000.0, clock)

    def check(self) -> None:
        observed = self.clock()
        if type(observed) not in (int, float) or not isfinite(float(observed)):
            raise ValueError("deadline clock must return a finite number")
        if float(observed) >= self.expires_at:
            raise TimeoutError("graph ranking deadline exceeded")


@dataclass(frozen=True, slots=True)
class PPRAlgorithmConfig:
    """Effective ``ppr_v1`` settings covered by the algorithm signature."""

    canonical_resolver_version: str
    rrf_k: int = 60
    max_seeds: int = 64
    max_scope_documents: int = 10_000
    max_scope_collections: int = 128
    max_hops: int = 2
    max_fanout: int = 10
    max_nodes: int = MAX_PPR_NODES
    max_edges: int = MAX_PPR_EDGES
    max_evidence_rows: int = 3_000
    max_evidence_per_edge: int = 3
    max_mentions_per_entity: int = 2
    ppr_restart: float = 0.20
    ppr_iterations: int = MAX_PPR_ITERATIONS
    max_candidates: int = 20
    max_per_document: int = 3
    timeout_ms: int = 150

    def __post_init__(self) -> None:
        version = self.canonical_resolver_version
        if (
            type(version) is not str
            or not version
            or version != version.strip()
            or len(version) > 128
            or "\x00" in version
        ):
            raise ValueError(
                "canonical_resolver_version must be a bounded nonempty exact string"
            )

        limits = {
            "rrf_k": 1_000,
            "max_seeds": 64,
            "max_scope_documents": 10_000,
            "max_scope_collections": 128,
            "max_hops": 2,
            "max_fanout": 10,
            "max_nodes": MAX_PPR_NODES,
            "max_edges": MAX_PPR_EDGES,
            "max_evidence_rows": 3_000,
            "max_evidence_per_edge": 3,
            "max_mentions_per_entity": 2,
            "ppr_iterations": MAX_PPR_ITERATIONS,
            "max_candidates": 20,
            "max_per_document": 3,
            "timeout_ms": 150,
        }
        for field_name, maximum in limits.items():
            value = getattr(self, field_name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(
                    f"{field_name} must be an exact integer in [1, {maximum}]"
                )
        try:
            restart = _restart_probability(self.ppr_restart)
        except ValueError as error:
            raise ValueError(
                "ppr_restart must be finite and strictly between 0 and 1"
            ) from error
        object.__setattr__(self, "ppr_restart", restart)

        if self.max_scope_collections > self.max_scope_documents:
            raise ValueError(
                "max_scope_collections must not exceed max_scope_documents"
            )
        if self.max_edges > self.max_nodes * self.max_fanout:
            raise ValueError("max_edges must not exceed max_nodes * max_fanout")
        if self.max_evidence_per_edge > self.max_evidence_rows:
            raise ValueError(
                "max_evidence_per_edge must not exceed the evidence row cap"
            )
        if self.max_per_document > self.max_candidates:
            raise ValueError("max_per_document must not exceed max_candidates")


class RetrievalDirection(StrEnum):
    """Direction of a relation transition used only for retrieval."""

    FORWARD = "forward"
    REVERSE_DIRECTED = "reverse_directed"
    UNDIRECTED = "undirected"


def _unit_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number in [0, 1]") from error
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return number


def _nonnegative_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _restart_probability(value: object) -> float:
    number = _unit_float(value, "restart_probability")
    if not 0.0 < number < 1.0:
        raise ValueError(
            "restart_probability must be finite and strictly between 0 and 1"
        )
    return number


def _stable_node_key(value: object) -> StableNodeKey:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("node key must be a stable canonical or local identity tuple")
    kind, identifier = value
    if kind == "canonical":
        if type(identifier) is not int or not 1 <= identifier <= 2**63 - 1:
            raise ValueError("canonical node key must contain a positive database ID")
    elif kind == "local":
        if type(identifier) is not str or not identifier or "\x00" in identifier:
            raise ValueError("local node key must contain a nonempty cluster key")
    else:
        raise ValueError("node key must be a stable canonical or local identity tuple")
    return cast(StableNodeKey, value)


def _safe_fsum(values: Iterable[float], label: str) -> float:
    try:
        total = fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} produced non-finite math") from error
    if not isfinite(total) or total < 0.0:
        raise ValueError(f"{label} produced non-finite math")
    return total


def transition_direction_factor(direction: RetrievalDirection) -> float:
    """Return the frozen retrieval multiplier for one relation direction."""

    if type(direction) is not RetrievalDirection:
        raise ValueError("direction must be an exact RetrievalDirection value")
    if direction is RetrievalDirection.REVERSE_DIRECTED:
        return REVERSE_DIRECTION_FACTOR
    return 1.0


def support_factor(support_count: int) -> float:
    """Apply the frozen, capped logarithmic evidence-support multiplier."""

    if type(support_count) is not int or support_count < 1:
        raise ValueError("support_count must be a positive integer")
    return 1.0 + log1p(min(support_count, SUPPORT_CAP)) / log1p(SUPPORT_CAP)


def utility_factor(destination_retrieval_utility: float) -> float:
    """Apply the frozen destination-utility floor and interpolation."""

    utility = _unit_float(
        destination_retrieval_utility, "destination retrieval utility"
    )
    return UTILITY_FLOOR + (1.0 - UTILITY_FLOOR) * utility


def raw_edge_weight(
    *,
    direction: RetrievalDirection,
    confidence: float,
    support_count: int,
    destination_retrieval_utility: float,
) -> float:
    """Compute one pre-normalized relation transition weight."""

    weight = (
        transition_direction_factor(direction)
        * _unit_float(confidence, "confidence")
        * support_factor(support_count)
        * utility_factor(destination_retrieval_utility)
    )
    if not isfinite(weight) or weight < 0.0:
        raise ValueError("raw edge weight must be finite and nonnegative")
    return weight


def normalize_transition_rows(
    weighted_rows: WeightedTransitionRows,
    *,
    nodes: Iterable[StableNodeKey] = (),
) -> NormalizedTransitionRows:
    """Combine targets and normalize each row in private stable-key order.

    Zero-weight edges are discarded. Sources, positive-weight targets, and
    explicit ``nodes`` all receive a row; an empty row denotes a dangling node.
    """

    if not isinstance(weighted_rows, Mapping):
        raise ValueError("transition rows must be a mapping")
    if len(weighted_rows) > MAX_PPR_NODES:
        raise ValueError(f"transition source rows exceed the {MAX_PPR_NODES} node cap")

    collected: dict[StableNodeKey, dict[StableNodeKey, list[float]]] = {}
    explicit_node_count = 0
    try:
        for raw_node in nodes:
            explicit_node_count += 1
            if explicit_node_count > MAX_PPR_NODES:
                raise ValueError(
                    f"explicit transition nodes exceed the {MAX_PPR_NODES} node cap"
                )
            node = _stable_node_key(raw_node)
            if node not in collected and len(collected) >= MAX_PPR_NODES:
                raise ValueError(
                    f"transition graph exceeds the {MAX_PPR_NODES} node cap"
                )
            collected.setdefault(node, {})
    except TypeError as error:
        raise ValueError("nodes must be an iterable of stable node keys") from error

    source_row_count = 0
    raw_edge_count = 0
    for raw_row in weighted_rows.items():
        source_row_count += 1
        if source_row_count > MAX_PPR_NODES:
            raise ValueError(
                f"transition source rows exceed the {MAX_PPR_NODES} source row cap"
            )
        if type(raw_row) is not tuple or len(raw_row) != 2:
            raise ValueError(
                "each transition source row must be an exact (source, edges) tuple"
            )
        raw_source, raw_edges = raw_row
        source = _stable_node_key(raw_source)
        if source not in collected and len(collected) >= MAX_PPR_NODES:
            raise ValueError(f"transition graph exceeds the {MAX_PPR_NODES} node cap")
        targets = collected.setdefault(source, {})
        try:
            edges = iter(raw_edges)
        except TypeError as error:
            raise ValueError("each transition row must be iterable") from error
        for raw_edge in edges:
            raw_edge_count += 1
            if raw_edge_count > MAX_PPR_EDGES:
                raise ValueError(
                    f"raw transition groups exceed the {MAX_PPR_EDGES} edge cap"
                )
            if type(raw_edge) is not tuple or len(raw_edge) != 2:
                raise ValueError(
                    "each transition must be an exact (target, weight) tuple"
                )
            raw_target, raw_weight = raw_edge
            target = _stable_node_key(raw_target)
            weight = _nonnegative_float(raw_weight, "transition weight")
            if weight == 0.0:
                continue
            if target not in collected and len(collected) >= MAX_PPR_NODES:
                raise ValueError(
                    f"transition graph exceeds the {MAX_PPR_NODES} node cap"
                )
            collected.setdefault(target, {})
            targets.setdefault(target, []).append(weight)

    normalized: NormalizedTransitionRows = {}
    for source in sorted(collected):
        combined: dict[StableNodeKey, float] = {}
        for target in sorted(collected[source]):
            combined[target] = _safe_fsum(
                sorted(collected[source][target]), "transition target sum"
            )
        row_total = _safe_fsum(
            (combined[target] for target in sorted(combined)), "transition row sum"
        )
        if row_total == 0.0:
            normalized[source] = ()
            continue
        row = tuple(
            (target, combined[target] / row_total) for target in sorted(combined)
        )
        if any(not isfinite(share) or share <= 0.0 for _, share in row):
            raise ValueError("transition normalization produced invalid math")
        normalized[source] = row
    return normalized


def _normalize_restart_vector(
    restart_weights: Mapping[StableNodeKey, int | float],
) -> dict[StableNodeKey, float]:
    if not isinstance(restart_weights, Mapping) or not restart_weights:
        raise ValueError("restart weights must be a nonempty mapping")
    if len(restart_weights) > MAX_PPR_NODES:
        raise ValueError(f"restart weights exceed the {MAX_PPR_NODES} node cap")
    validated: dict[StableNodeKey, float] = {}
    restart_count = 0
    for raw_node, raw_weight in restart_weights.items():
        restart_count += 1
        if restart_count > MAX_PPR_NODES:
            raise ValueError(f"restart weights exceed the {MAX_PPR_NODES} node cap")
        node = _stable_node_key(raw_node)
        validated[node] = _nonnegative_float(raw_weight, "restart weight")
    total = _safe_fsum(
        (validated[node] for node in sorted(validated)), "restart weight sum"
    )
    if total == 0.0:
        raise ValueError("restart weights must have a positive sum")
    normalized = {node: validated[node] / total for node in sorted(validated)}
    if any(not isfinite(weight) or weight < 0.0 for weight in normalized.values()):
        raise ValueError("restart normalization produced invalid math")
    return normalized


def personalized_pagerank(
    restart_weights: Mapping[StableNodeKey, int | float],
    transition_rows: WeightedTransitionRows,
    *,
    restart_probability: float,
    iterations: int,
    _deadline: _MonotonicDeadline | None = None,
) -> dict[StableNodeKey, float]:
    """Run the exact fixed-iteration ``ppr_v1`` recurrence.

    The supplied transition weights are normalized deterministically. The
    restart weights are likewise normalized once and define ``p(0)``.
    """

    alpha = _restart_probability(restart_probability)
    if (
        type(iterations) is not int
        or not 1 <= iterations <= MAX_PPR_ITERATIONS
    ):
        raise ValueError(
            f"iterations must be an exact integer in [1, {MAX_PPR_ITERATIONS}]"
        )
    if _deadline is not None and type(_deadline) is not _MonotonicDeadline:
        raise ValueError("_deadline must be the private monotonic deadline")

    restart = _normalize_restart_vector(restart_weights)
    matrix = normalize_transition_rows(transition_rows, nodes=restart)
    ordered_nodes = tuple(matrix)
    restart = {node: restart.get(node, 0.0) for node in ordered_nodes}
    scores = dict(restart)
    damping = 1.0 - alpha

    for _ in range(iterations):
        if _deadline is not None:
            _deadline.check()
        incoming: dict[StableNodeKey, list[float]] = defaultdict(list)
        dangling_scores: list[float] = []
        for source in ordered_nodes:
            row = matrix[source]
            if not row:
                dangling_scores.append(scores[source])
                continue
            for target, share in row:
                incoming[target].append(scores[source] * share)
        dangling_mass = _safe_fsum(dangling_scores, "dangling mass")

        next_scores: dict[StableNodeKey, float] = {}
        for target in ordered_nodes:
            propagated = _safe_fsum(incoming[target], "incoming transition mass")
            restart_mass = restart[target]
            score = alpha * restart_mass + damping * (
                propagated + dangling_mass * restart_mass
            )
            if not isfinite(score) or score < 0.0:
                raise ValueError("PPR iteration produced non-finite math")
            next_scores[target] = score
        scores = next_scores
        if _deadline is not None:
            _deadline.check()
    return scores


def edge_evidence_flow(
    *,
    restart_probability: float,
    source_score: float,
    normalized_share: float,
) -> float:
    """Return ``(1 - alpha) * p[source] * normalized_group_share``."""

    alpha = _restart_probability(restart_probability)
    score = _unit_float(source_score, "source score")
    share = _unit_float(normalized_share, "normalized share")
    flow = (1.0 - alpha) * score * share
    if not isfinite(flow) or flow < 0.0:
        raise ValueError("edge evidence flow produced non-finite math")
    return flow


def _canonical_decimal(value: float) -> str:
    rendered = format(Decimal(str(value)).normalize(), "f")
    return "0" if rendered in {"-0", ""} else rendered


def canonical_algorithm_json(config: PPRAlgorithmConfig) -> bytes:
    """Serialize every effective ``ppr_v1`` setting as canonical UTF-8 JSON."""

    if type(config) is not PPRAlgorithmConfig:
        raise ValueError("config must be an exact PPRAlgorithmConfig value")
    payload: dict[str, int | str] = {
        "algorithm": ALGORITHM_VERSION,
        "canonical_resolver_version": config.canonical_resolver_version,
        "evidence_version": EVIDENCE_VERSION,
        "max_candidates": config.max_candidates,
        "max_edges": config.max_edges,
        "max_evidence_per_edge": config.max_evidence_per_edge,
        "max_evidence_rows": config.max_evidence_rows,
        "max_fanout": config.max_fanout,
        "max_hops": config.max_hops,
        "max_mentions_per_entity": config.max_mentions_per_entity,
        "max_nodes": config.max_nodes,
        "max_per_document": config.max_per_document,
        "max_scope_collections": config.max_scope_collections,
        "max_scope_documents": config.max_scope_documents,
        "max_seeds": config.max_seeds,
        "mention_factor": _canonical_decimal(MENTION_FACTOR),
        "ppr_iterations": config.ppr_iterations,
        "ppr_restart": _canonical_decimal(config.ppr_restart),
        "reverse_factor": _canonical_decimal(REVERSE_DIRECTION_FACTOR),
        "rrf_k": config.rrf_k,
        "seed_version": SEED_VERSION,
        "support_cap": SUPPORT_CAP,
        "timeout_ms": config.timeout_ms,
        "transition_version": TRANSITION_VERSION,
        "utility_floor": _canonical_decimal(UTILITY_FLOOR),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def graph_algorithm_signature(config: PPRAlgorithmConfig) -> str:
    """Return the lowercase SHA-256 covering the effective algorithm config."""

    return sha256(canonical_algorithm_json(config)).hexdigest()


__all__ = [
    "ALGORITHM_VERSION",
    "EVIDENCE_VERSION",
    "MENTION_FACTOR",
    "MAX_PPR_EDGES",
    "MAX_PPR_ITERATIONS",
    "MAX_PPR_NODES",
    "NormalizedTransitionRows",
    "PPRAlgorithmConfig",
    "REVERSE_DIRECTION_FACTOR",
    "RetrievalDirection",
    "SEED_VERSION",
    "StableNodeKey",
    "SUPPORT_CAP",
    "TRANSITION_VERSION",
    "UTILITY_FLOOR",
    "WeightedTransitionRows",
    "canonical_algorithm_json",
    "edge_evidence_flow",
    "graph_algorithm_signature",
    "normalize_transition_rows",
    "personalized_pagerank",
    "raw_edge_weight",
    "support_factor",
    "transition_direction_factor",
    "utility_factor",
]
