"""Production retrieval execution and privacy-safe Task21 observation traces."""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from apps.documents.services.hybrid_graph_authorization import (
    HybridGraphRetrievalDependencies,
)

from . import task21_hybrid_live_trace_build as _trace


class RecordingTopologyLoader:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.calls: list[tuple[object, tuple[object, ...], object, object]] = []

    def load(self, *, ready, seeds, caps, deadline):
        snapshot = self.delegate.load(
            ready=ready, seeds=seeds, caps=caps, deadline=deadline
        )
        self.calls.append((ready, seeds, caps, snapshot))
        return snapshot


class RecordingRuntime:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.shared = None
        self.direct = None
        self.extended = None

    def prepare_shared(self, **kwargs):
        self.shared = self.delegate.prepare_shared(**kwargs)
        return self.shared

    def run_direct(self, **kwargs):
        self.direct = self.delegate.run_direct(**kwargs)
        return self.direct

    def prepare_extended(self, **kwargs):
        return self.delegate.prepare_extended(**kwargs)

    def run_extended(self, **kwargs):
        self.extended = self.delegate.run_extended(**kwargs)
        return self.extended


class RecordingMaterializer:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.by_key: dict[str, int] = {}

    def __call__(self, **kwargs):
        rows = self.delegate(**kwargs)
        for row in rows:
            previous = self.by_key.setdefault(row.chunk_key, row.integer_chunk_pk)
            if previous != row.integer_chunk_pk:
                raise RuntimeError("projected chunk reversal drifted")
        return rows


@dataclass(frozen=True, slots=True)
class GraphExecutionTrace:
    runtime: RecordingRuntime
    topology: RecordingTopologyLoader
    materializer: RecordingMaterializer
    pool: tuple[object, ...]
    graph_ms: float


def _dependencies(authorization, settings) -> HybridGraphRetrievalDependencies:
    from apps.documents.services.hybrid_graph_dependencies import (
        build_hybrid_graph_dependencies,
    )

    base = build_hybrid_graph_dependencies(
        authorization=authorization, settings=settings
    )
    if type(base) is not HybridGraphRetrievalDependencies:
        raise RuntimeError("production graph dependencies are unavailable")
    topology = RecordingTopologyLoader(base.runtime.topology_loader)
    base.runtime.topology_loader = topology
    runtime = RecordingRuntime(base.runtime)
    materializer = RecordingMaterializer(base.materialize)
    return HybridGraphRetrievalDependencies(runtime, settings, materializer)


def _graph_pool(prepared, settings) -> GraphExecutionTrace:
    from apps.documents.services.hybrid_graph_orchestration import (
        hybrid_graph_candidate_pool,
    )

    dependencies = _dependencies(prepared.authorization, settings)
    started = perf_counter()
    pool, diagnostics = hybrid_graph_candidate_pool(
        prepared.snapshot,
        prepared.case["query"],
        prepared.authorization,
        dependencies,
    )
    elapsed = (perf_counter() - started) * 1_000
    if diagnostics.get("graph_status") == "error":
        raise RuntimeError("production graph branch failed closed")
    return GraphExecutionTrace(
        dependencies.runtime,
        dependencies.runtime.delegate.topology_loader,
        dependencies.materialize,
        tuple(pool),
        elapsed,
    )


def _ranked_pool(prepared, pool, reranker_calls: int):
    if reranker_calls == 0:
        return tuple(pool[:10]), 0.0, 0
    from apps.documents.models.chunks import TextChunk
    from apps.documents.services.chunk_rerank import _STRICT_EVALUATION_RERANK
    from apps.documents.services.chunk_search import materialize_and_rerank_candidates

    ranked = materialize_and_rerank_candidates(
        TextChunk,
        prepared.case["query"],
        10,
        tuple(pool),
        authorized_scope=None,
        force_complete_rerank=True,
        _eval_rerank_capability=_STRICT_EVALUATION_RERANK,
    )
    return (
        ranked.ranked_results,
        ranked.rerank_ms,
        ranked.inaccessible_candidate_count,
    )


class ProductionArmExecutor:
    def __init__(self, base_settings) -> None:
        self.base_settings = base_settings
        self.parity_calls: list[tuple[object, object]] = []
        self.ready_scopes: list[object] = []

    def run_arm(self, *, case, prepared, spec):
        started = perf_counter()
        graph_enabled = spec["direct_enabled"] or spec["extended_enabled"]
        first = second = None
        pool = prepared.snapshot.baseline_candidates
        branch_ms = 0.0
        if graph_enabled:
            settings = replace(
                self.base_settings,
                graph_direct_enabled=spec["direct_enabled"],
                graph_extended_enabled=spec["extended_enabled"],
            )
            first, second = _graph_pool(prepared, settings), _graph_pool(
                prepared, settings
            )
            pool, branch_ms = first.pool, first.graph_ms
            self.parity_calls.extend(zip(first.topology.calls, second.topology.calls))
            for trace in (first, second):
                if trace.runtime.shared is not None:
                    self.ready_scopes.append(trace.runtime.shared.scope)
        ranked, rerank_ms, inaccessible_count = _ranked_pool(
            prepared, pool, spec["reranker_calls"]
        )
        symbols = prepared.chunk_symbols_by_pk
        ranked_symbols = tuple(symbols[row.pk] for row in ranked)
        projected: tuple[str, ...] = ()
        repeated: tuple[str, ...] = ()
        branch_maps = {"direct": {}, "extended": {}}
        if first is not None and second is not None:
            projected, branch_maps = _trace.projected_symbols(first, symbols)
            repeated, _unused = _trace.projected_symbols(second, symbols)
            if projected != repeated:
                raise RuntimeError("projected graph ranks are nondeterministic")
        expected_arms = set(case.get("task21_expected_arms", ()))
        if spec["name"] in expected_arms and not projected:
            raise RuntimeError("expected graph arm produced no projected candidates")
        baseline_ranks = {
            row.pk: rank
            for rank, row in enumerate(prepared.snapshot.baseline_candidates, 1)
        }
        rerank_ranks = {row.pk: rank for rank, row in enumerate(ranked, 1)}
        candidate_trace = []
        for ordinal, row in enumerate(pool, 1):
            direct = branch_maps["direct"].get(row.pk)
            extended = branch_maps["extended"].get(row.pk)
            candidate_trace.append(
                {
                    "chunk_id": symbols[row.pk],
                    "ordinal": ordinal,
                    "sources": [
                        name
                        for name, present in (
                            ("baseline", row.pk in baseline_ranks),
                            ("direct", direct is not None),
                            ("extended", extended is not None),
                        )
                        if present
                    ],
                    "baseline_rank": baseline_ranks.get(row.pk),
                    "direct_rank": direct[0] if direct else None,
                    "direct_score_hex": direct[1].hex() if direct else None,
                    "extended_rank": extended[0] if extended else None,
                    "extended_score_hex": extended[1].hex() if extended else None,
                    "fusion_score_hex": sum(
                        (
                            1.0 / (60 + value[0])
                            for value in (direct, extended)
                            if value is not None
                        ),
                        0.0,
                    ).hex(),
                    "reranker_rank": rerank_ranks.get(row.pk),
                }
            )
        seed_symbols = (
            tuple(symbols[row.chunk_id] for row in prepared.snapshot.graph_seeds)
            if spec["extended_enabled"]
            else ()
        )
        mapped_seeds = (
            _trace.mapped_seed_symbols(first, prepared)
            if spec["extended_enabled"] and first is not None
            else ()
        )
        graph_symbols = tuple(item for item in ranked_symbols if item in projected)
        inaccessible = tuple(
            item
            for item in ranked_symbols
            if item not in prepared.accessible_chunk_symbols
        )
        if inaccessible:
            raise RuntimeError("live retrieval returned case-inaccessible evidence")
        total_ms = (perf_counter() - started) * 1_000
        candidate_ms = sum(
            (
                prepared.snapshot.vector_ms,
                prepared.snapshot.trigram_ms,
                prepared.snapshot.exact_ms,
            )
        )
        return {
            "case_id": case["id"],
            "ranked_chunk_ids": list(ranked_symbols),
            "graph_chunk_ids": list(graph_symbols),
            "citation_evidence_chunk_ids": list(ranked_symbols),
            "seed_chunk_ids": list(seed_symbols),
            "mapped_seed_chunk_ids": list(mapped_seeds),
            "projected_ranks": list(projected),
            "repeated_projected_ranks": list(repeated),
            "adversarial_candidate_chunk_ids": list(prepared.adversarial_chunk_symbols),
            "inaccessible_result_chunk_ids": [],
            "latency_ms": total_ms,
            "reranker_calls": spec["reranker_calls"],
            "comparison_snapshot_signature": _trace.comparison_signature(prepared),
            "candidate_trace": candidate_trace,
            "timing_trace": {
                "candidate_ms": candidate_ms,
                "branch_ms": branch_ms,
                "fusion_ms": max(0.0, total_ms - branch_ms - rerank_ms),
                "rerank_ms": rerank_ms,
                "total_ms": total_ms,
            },
            "authorization_status": "current",
            "graph_scheduled": bool(graph_enabled),
            "inaccessible_candidate_count": inaccessible_count,
        }


__all__ = ["ProductionArmExecutor", "RecordingTopologyLoader"]
