"""Concurrent graph branches, authorization revalidation, and deterministic fusion."""

from __future__ import annotations

from math import isfinite
from time import perf_counter

from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.documents.services.chunk_search_legacy_graph import graph_diagnostics
from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
    reauthorized_baseline,
)


def _candidate_identifier(candidate: object) -> int:
    identifier = getattr(candidate, "pk", None)
    if type(identifier) is not int or identifier <= 0:
        raise ValueError("candidate must expose a positive integer primary key")
    return identifier


def hybrid_graph_candidate_pool(
    snapshot: HybridCandidateSnapshot,
    query: str,
    authorization: object,
    dependencies: HybridGraphRetrievalDependencies,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Run both projected branches, reauthorize, and fuse without reranking."""

    from apps.documents.services.chunk_search_fusion import (
        BaselineCandidate,
        CandidateSource,
        GraphBranchInput,
        GraphCandidate,
        fuse_candidates,
        graph_candidate_order_checksum,
    )
    from apps.knowledge_graph.projection.identifiers import (
        OpaqueProjectionKey,
        ProjectionIdentifierDomain,
    )
    from apps.knowledge_graph.retrieval.branch_contracts import BranchStatusV1
    from apps.knowledge_graph.retrieval.materialization import (
        MaterializedGraphChunkV1,
    )
    from apps.knowledge_graph.retrieval.scheduler import run_hybrid_graph_branches

    started = perf_counter()
    baseline = snapshot.baseline_candidates

    def diagnostics(status: str, candidate_count: int = 0) -> dict[str, object]:
        return graph_diagnostics(
            started_at=started,
            seed_count=len(snapshot.graph_seeds),
            candidate_count=candidate_count,
            status=status,
            algorithm_signature=None,
            version_signature=None,
        )

    settings = dependencies.settings
    traversal_enabled = getattr(settings, "memgraph_traversal_enabled", None)
    direct_enabled = getattr(settings, "graph_direct_enabled", None)
    extended_enabled = getattr(settings, "graph_extended_enabled", None)
    if (
        traversal_enabled is not True
        or type(direct_enabled) is not bool
        or type(extended_enabled) is not bool
        or not (direct_enabled or extended_enabled)
    ):
        return baseline, diagnostics("miss")
    overall_timeout_ms = getattr(settings, "graph_overall_timeout_ms", None)
    if type(overall_timeout_ms) is not int or not 1 <= overall_timeout_ms <= 5_000:
        return baseline, diagnostics("error")
    deadline = perf_counter() + overall_timeout_ms / 1_000.0
    if not isfinite(deadline):
        return baseline, diagnostics("error")
    try:
        outcome = run_hybrid_graph_branches(
            runtime=dependencies.runtime,
            query=query,
            baseline=snapshot,
            authorization=authorization,
            settings=settings,
            deadline=deadline,
        )
    except Exception:
        try:
            current_baseline, _graph_allowed = reauthorized_baseline(
                baseline, authorization
            )
        except Exception:
            current_baseline = ()
        return current_baseline, diagnostics("error")

    try:
        current_baseline, graph_allowed = reauthorized_baseline(baseline, authorization)
    except Exception:
        return (), diagnostics("error")
    if not graph_allowed:
        return current_baseline, diagnostics("error")
    if outcome.shared_failure_reason is not None:
        return current_baseline, diagnostics("error")

    successful = tuple(
        envelope
        for envelope in (outcome.direct, outcome.extended)
        if envelope.status is BranchStatusV1.SUCCEEDED
    )
    raw_candidates = tuple(
        candidate
        for envelope in successful
        for candidate in envelope.result.candidates  # type: ignore[union-attr]
    )
    chunk_key_values = tuple(sorted({row.chunk_key for row in raw_candidates}))
    if not chunk_key_values:
        return current_baseline, diagnostics("miss")
    chunk_keys = tuple(
        OpaqueProjectionKey(ProjectionIdentifierDomain.CHUNK, value)
        for value in chunk_key_values
    )
    try:
        materialized = dependencies.materialize(
            chunk_keys=chunk_keys,
            authorization=authorization,
            outcome=outcome,
        )
        if type(materialized) is not tuple or any(
            type(row) is not MaterializedGraphChunkV1 for row in materialized
        ):
            raise TypeError("graph materializer returned invalid rows")
        by_key = {row.chunk_key: row for row in materialized}
        if len(by_key) != len(materialized) or set(by_key) != set(chunk_key_values):
            raise ValueError("graph materialization did not cover the exact pool")
    except Exception:
        return current_baseline, diagnostics("error")

    try:
        final_baseline, graph_allowed = reauthorized_baseline(
            current_baseline, authorization
        )
    except Exception:
        return (), diagnostics("error")
    if not graph_allowed:
        return final_baseline, diagnostics("error")

    try:
        baseline_rows = tuple(
            BaselineCandidate(
                _candidate_identifier(row),
                getattr(row, "doc_id"),
                getattr(row, "chunk_number"),
                row,
            )
            for row in final_baseline
        )
        ready_checksums = {
            envelope.result.provenance.ready_bundle_checksum  # type: ignore[union-attr]
            for envelope in successful
        }
        if len(ready_checksums) != 1:
            raise ValueError("branch ready bundles disagree")
        ready_checksum = ready_checksums.pop()

        def branch_input(envelope, source):
            if envelope.status is not BranchStatusV1.SUCCEEDED:
                return None
            result = envelope.result
            rows = tuple(
                GraphCandidate(
                    by_key[row.chunk_key].integer_chunk_pk,
                    by_key[row.chunk_key].document_uuid,
                    by_key[row.chunk_key].chunk_number,
                    by_key[row.chunk_key].candidate_object,
                    row.chunk_key,
                    row.rank,
                    row.score,
                )
                for row in result.candidates
            )
            return GraphBranchInput(
                source,
                result.provenance.ready_bundle_checksum,
                len(rows),
                graph_candidate_order_checksum(rows),
                rows,
            )

        direct = branch_input(outcome.direct, CandidateSource.DIRECT)
        extended = branch_input(outcome.extended, CandidateSource.EXTENDED)
        direct_cap = int(getattr(settings, "graph_direct_max_candidates"))
        extended_cap = int(getattr(settings, "graph_extended_max_candidates"))
        fused = fuse_candidates(
            baseline=baseline_rows,
            direct=direct,
            extended=extended,
            expected_ready_bundle_checksum=ready_checksum,
            direct_cap=direct_cap,
            extended_cap=extended_cap,
            graph_cap=max(direct_cap, extended_cap),
            rrf_k=int(getattr(settings, "graph_fusion_rrf_k")),
        )
        if fused.diagnostics.malformed_provenance:
            return final_baseline, diagnostics("error")
        graph_count = fused.diagnostics.graph_only_selected
        return fused.rerank_candidates, diagnostics(
            "hit" if graph_count else "miss", graph_count
        )
    except Exception:
        return final_baseline, diagnostics("error")


__all__ = ["hybrid_graph_candidate_pool"]
