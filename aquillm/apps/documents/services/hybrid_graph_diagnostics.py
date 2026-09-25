"""Fixed-label aggregate branch diagnostics; never serialize source objects."""

import structlog

logger = structlog.stdlib.get_logger(__name__)


class GraphBranchDiagnostics:
    def __init__(self):
        self.outcome = None
        self.materialized = ()
        self.baseline = ()
        self.fusion = None

    def emit(self):
        from apps.knowledge_graph.retrieval.branch_contracts import BranchStatusV1

        fields = {}
        by_key = {row.chunk_key: row.integer_chunk_pk for row in self.materialized}
        baseline_ids = {row.pk for row in self.baseline}
        raw_count = 0
        for name in ("direct", "extended"):
            envelope = getattr(self.outcome, name, None)
            status, reason, raw, elapsed, duplicates, new = (
                "not_run",
                "not_run",
                0,
                0,
                0,
                0,
            )
            if envelope is not None:
                elapsed = envelope.diagnostics.elapsed_ms
                if envelope.status is BranchStatusV1.FAILED:
                    status, reason = "failed", envelope.failure_reason.value
                else:
                    candidates = envelope.result.candidates
                    raw = len(candidates)
                    duplicates = sum(
                        by_key.get(row.chunk_key) in baseline_ids for row in candidates
                    )
                    new = sum(
                        row.chunk_key in by_key
                        and by_key[row.chunk_key] not in baseline_ids
                        for row in candidates
                    )
                    status = (
                        "succeeded_new"
                        if new
                        else "succeeded_duplicates"
                        if duplicates
                        else "succeeded_empty"
                        if not raw
                        else "succeeded_unmaterialized"
                    )
                    reason = "none"
            raw_count += raw
            fields.update(
                {
                    f"graph_{name}_status": status,
                    f"graph_{name}_reason": reason,
                    f"graph_{name}_raw_count": raw,
                    f"graph_{name}_duplicate_count": duplicates,
                    f"graph_{name}_new_count": new,
                    f"graph_{name}_elapsed_ms": elapsed,
                }
            )
        fields.update(
            graph_raw_count=raw_count,
            graph_materialized_count=len(self.materialized),
            graph_baseline_duplicate_count=getattr(
                self.fusion, "baseline_duplicate_count", 0
            ),
            graph_cross_branch_duplicate_count=getattr(
                self.fusion, "cross_branch_duplicate_count", 0
            ),
            graph_new_count=getattr(self.fusion, "graph_only_selected", 0),
        )
        logger.info("obs.rag.graph_branches", **fields)
        return fields
