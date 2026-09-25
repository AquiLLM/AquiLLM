"""PostgreSQL oracle versus live Memgraph parity for Task21 topology calls."""

from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _aggregate(values: list[str]) -> str:
    return sha256(_canonical(values)).hexdigest()


def _score_bytes(result) -> bytes:
    return _canonical([[key, value.hex()] for key, value in result.scores])


def _tie_bytes(result) -> bytes:
    by_score: dict[str, list[str]] = defaultdict(list)
    for key, score in result.scores:
        by_score[score.hex()].append(key)
    return _canonical(
        [
            [score, sorted(keys)]
            for score, keys in sorted(by_score.items())
            if len(keys) > 1
        ]
    )


def canonical_comparison_inputs(rows) -> list[dict[str, object]]:
    """Select the first deterministic parity input observed for each branch."""

    by_branch: dict[str, dict[str, object]] = {}
    for raw in rows:
        row = dict(raw)
        branch = row.get("branch")
        if branch not in {"direct", "extended"}:
            raise RuntimeError("live parity branch is not exact")
        by_branch.setdefault(branch, row)
    if set(by_branch) != {"direct", "extended"}:
        raise RuntimeError("live parity inputs require direct and extended")
    return [by_branch[branch] for branch in ("direct", "extended")]


def build_live_backend_parity(*, call_pairs, ready_scopes, settings):
    """Replay each exact live provider input through the PostgreSQL oracle."""

    from apps.knowledge_graph.projection.django_projection_source import (
        DjangoProjectionRowSource,
    )
    from apps.knowledge_graph.projection.postgres_repository import (
        PostgresProjectionRepository,
    )
    from apps.knowledge_graph.projection.topology_snapshot import (
        build_projected_topology_snapshot,
    )
    from apps.knowledge_graph.retrieval.production_runtime_support import ppr_config
    from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
    from apps.knowledge_graph.retrieval.projected_types import (
        canonical_projected_snapshot_bytes,
    )
    from apps.knowledge_graph.retrieval.topology.contracts import (
        projected_seed_checksum,
    )

    pairs = tuple(call_pairs)
    if not pairs:
        raise RuntimeError("live parity has no Memgraph topology calls")
    scopes = {scope.ready.bundle_checksum: scope for scope in ready_scopes}
    source = DjangoProjectionRowSource(
        "projection_source",
        state_using="projection_source",
        identifier_key=(
            settings.projection_identifier_hmac_key.get_secret_value().encode()
        ),
        identifier_key_version=settings.projection_identifier_key_version,
        schema_version=settings.projection_schema_version,
        projection_version=settings.projection_format_version,
    )
    repository = PostgresProjectionRepository(
        using="projection_source", source=source
    )
    hashes = {
        backend: {kind: [] for kind in ("snapshot", "scores", "trace", "ties")}
        for backend in ("postgres", "memgraph")
    }
    comparisons = []
    first_ranks = None
    for first, repeated in pairs:
        ready, seeds, caps, memgraph = first
        if repeated[:3] != first[:3]:
            raise RuntimeError("repeated live topology input drifted")
        if canonical_projected_snapshot_bytes(repeated[3]) != (
            canonical_projected_snapshot_bytes(memgraph)
        ):
            raise RuntimeError("repeated Memgraph topology snapshot drifted")
        scope = scopes.get(ready.bundle_checksum)
        if scope is None:
            raise RuntimeError("live parity ready scope is unavailable")
        bundles = tuple(
            repository.load_projection_bundle(
                projection_id=authority.projection_id,
                batch_size=settings.projection_batch_size,
                purpose="audit",
            )
            for authority in scope.projections
        )
        postgres = build_projected_topology_snapshot(
            ready=ready, seeds=seeds, caps=caps, bundles=bundles
        )
        snapshots = {
            "postgres": canonical_projected_snapshot_bytes(postgres),
            "memgraph": canonical_projected_snapshot_bytes(memgraph),
        }
        if snapshots["postgres"] != snapshots["memgraph"]:
            raise RuntimeError("PostgreSQL/Memgraph snapshot parity failed")
        maximum = (
            caps.max_results
            if hasattr(caps, "max_results")
            else caps.max_candidates
        )
        results = {
            "postgres": ppr_projected_v1(
                snapshot=postgres,
                seeds=seeds,
                config=ppr_config(postgres, maximum),
            ),
            "memgraph": ppr_projected_v1(
                snapshot=memgraph,
                seeds=seeds,
                config=ppr_config(memgraph, maximum),
            ),
        }
        if results["postgres"] != results["memgraph"]:
            raise RuntimeError("PostgreSQL/Memgraph PPR parity failed")
        if first_ranks is None:
            first_ranks = results["postgres"].ranked_identity_keys
        for backend in hashes:
            result = results[backend]
            values = {
                "snapshot": snapshots[backend],
                "scores": _score_bytes(result),
                "trace": result.trace_bytes,
                "ties": _tie_bytes(result),
            }
            for kind, encoded in values.items():
                hashes[backend][kind].append(sha256(encoded).hexdigest())
        comparisons.append(
            {
                "branch": caps.branch_kind.value,
                "ready_bundle_checksum": ready.bundle_checksum,
                "seed_checksum": projected_seed_checksum(seeds),
                "seed_count": len(seeds),
                "max_depth": caps.max_depth,
                "max_nodes": caps.max_nodes,
                "max_edges": caps.max_edges,
                "max_results": maximum,
                "projection_keys": [
                    row.projection_key for row in ready.selected_generations
                ],
                "generation_keys": [
                    row.generation_key for row in ready.selected_generations
                ],
                "authorized_document_keys": [
                    row.document_key for row in ready.authorized_documents
                ],
            }
        )
    if not first_ranks:
        raise RuntimeError("live parity PPR ranks are empty")
    result = {
        f"{backend}_{kind}_sha256": _aggregate(hashes[backend][kind])
        for backend in ("postgres", "memgraph")
        for kind in ("snapshot", "scores", "trace", "ties")
    }
    result.update(
        postgres_projected_ranks=list(first_ranks),
        memgraph_projected_ranks=list(first_ranks),
        comparison_inputs=canonical_comparison_inputs(comparisons),
    )
    return result


__all__ = ["build_live_backend_parity", "canonical_comparison_inputs"]
