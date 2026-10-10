"""One original baseline prefix and one deterministic seed allowance across groups."""

from dataclasses import replace
from math import fsum, isfinite

from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.chunk_search_scoped_rows import validate_rows

from .graph_runtime_seeds_v2 import group_seeds
from .topology.contracts import ProjectedSeedV1
from .types import GraphExpansionSeed


def validated_seed_baseline(baseline, *, authorization, settings):
    if not settings.graph_extended_enabled:
        return None, set()
    scope = current_scope()
    if (
        type(baseline) is not HybridCandidateSnapshot
        or scope is None
        or scope.authorization != authorization
        or {d.id for d in baseline.documents} != set(scope.requested_document_ids)
        or type(baseline.graph_seeds) is not tuple
        or baseline.graph_seed_error
    ):
        return None, {"not_ready"}
    rows = validate_rows(baseline.baseline_candidates, scope=scope)
    by_id = {r.pk: r for r in rows}
    if rows != baseline.baseline_candidates or any(
        r.doc_id not in scope.requested_document_ids for r in rows
    ):
        return None, {"not_ready"}
    seen, ranks = set(), set()
    for seed in baseline.graph_seeds:
        if (
            type(seed) is not GraphExpansionSeed
            or type(seed.chunk_id) is not int
            or seed.chunk_id not in by_id
            or seed.chunk_id in seen
            or type(seed.rank) is not int
            or seed.rank < 1
            or seed.rank in ranks
            or type(seed.restart_weight) is not float
            or not isfinite(seed.restart_weight)
            or seed.restart_weight <= 0
        ):
            return None, {"not_ready"}
        seen.add(seed.chunk_id)
        ranks.add(seed.rank)
    limit = settings.graph_extended_max_seeds
    prefix = baseline.graph_seeds[:limit]
    return replace(baseline, graph_seeds=prefix), (
        {"frontier_budget"} if len(prefix) < len(baseline.graph_seeds) else set()
    )


def group_baseline_view(baseline, group):
    if baseline is None:
        return None
    allowed = {
        r.pk
        for r in baseline.baseline_candidates
        if r.doc_id in group.scope.selected_document_ids
    }
    return replace(
        baseline,
        graph_seeds=tuple(
            seed for seed in baseline.graph_seeds if seed.chunk_id in allowed
        ),
    )


def allocate_group_seeds_v2(
    *,
    groups,
    runtime,
    query,
    baseline,
    authorization,
    settings,
    caps,
    deadline,
    frontier,
):
    from apps.collections.services.bounded_retrieval_authorization import (
        bounded_reauthorization_reads,
    )

    with bounded_reauthorization_reads(authorization, frontier):
        return _allocate(
            groups=groups,
            runtime=runtime,
            query=query,
            baseline=baseline,
            authorization=authorization,
            settings=settings,
            caps=caps,
            deadline=deadline,
            frontier=frontier,
        )


def _allocate(
    *,
    groups,
    runtime,
    query,
    baseline,
    authorization,
    settings,
    caps,
    deadline,
    frontier,
):
    baseline, reasons = validated_seed_baseline(
        baseline, authorization=authorization, settings=settings
    )
    prepared, local_reasons = {}, {}
    for group in groups:
        frontier.check_active()
        seeds, local = group_seeds(
            runtime=runtime,
            group=group,
            query=query,
            baseline=group_baseline_view(baseline, group),
            authorization=authorization,
            settings=settings,
            caps=caps,
            deadline=deadline,
            frontier=frontier,
        )
        prepared[group.group_checksum] = seeds
        local_reasons[group.group_checksum] = local
    nonempty = sum(bool(seeds) for seeds in prepared.values())
    pool = sorted(
        (seed.mass / nonempty, seed.identity_key, group)
        for group, seeds in prepared.items()
        for seed in seeds
    )
    pool.sort(key=lambda row: (-row[0], row[1], row[2]))
    if len(pool) > caps.max_seeds:
        reasons.add("frontier_budget")
    # Admit one best identity per nonempty group before any group's next identity.
    # This is a global allowance, with deterministic priority within each round.
    queues = {group: [row for row in pool if row[2] == group] for group in prepared}
    admitted = []
    while len(admitted) < min(caps.max_seeds, len(pool)):
        heads = sorted(
            (rows.pop(0) for rows in queues.values() if rows),
            key=lambda row: (-row[0], row[1], row[2]),
        )
        admitted.extend(heads[: caps.max_seeds - len(admitted)])
    total = fsum(mass for mass, _, _ in admitted)
    result = {}
    for group in prepared:
        rows = [(mass, key) for mass, key, owner in admitted if owner == group]
        group_mass = fsum(mass for mass, _ in rows)
        seeds = tuple(
            ProjectedSeedV1(key, mass / group_mass)
            for mass, key in sorted(rows, key=lambda row: row[1])
        )
        result[group] = (
            seeds,
            group_mass / total if total else 0.0,
            local_reasons[group],
        )
    return result, reasons
