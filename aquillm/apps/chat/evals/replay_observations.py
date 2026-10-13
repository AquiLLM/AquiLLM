"""Redacted replay reports from private in-memory retrieval events."""

import hashlib


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(row):
    return row.get("chunk_id", row.get("id", row.get("i")))


def _text(row):
    return row.get("text") or row.get("x") or row.get("content") or ""


def _safe_graph_reasons(reasons):
    from apps.knowledge_graph.retrieval.branch_contracts import (
        DirectBranchFailureReason,
        ExtendedBranchFailureReason,
        SharedBranchFailureReason,
    )

    if type(reasons) not in (list, tuple):
        return None
    allowed = {
        reason.value
        for enum in (
            DirectBranchFailureReason,
            ExtendedBranchFailureReason,
            SharedBranchFailureReason,
        )
        for reason in enum
    }
    # Exact strings only: never stringify arbitrary diagnostics or exceptions.
    return [
        reason for reason in reasons if type(reason) is str and reason in allowed
    ] or None


def summarize(events, packet, targets):
    stages = [data for name, data in events if name == "retrieval_stages"]
    result = {
        "stages": [],
        "targets": [],
        "packet": [
            {
                "text_sha256": fingerprint(_text(row)),
                "characters": len(_text(row)),
                "rank": rank,
            }
            for rank, row in enumerate(packet, 1)
        ],
        "graph_stage_timings": {"status": "unavailable"},
        "timings_ms": {},
        "budget": {"limits": None, "observed": None, "completion": None},
        "proposals": {"query_count": None},
    }
    for stage in stages:
        graph = stage.get("graph", {})
        result["stages"].append(
            {
                "effective_limits": stage.get("effective_limits"),
                "pool_count": len(stage.get("materialized_union", [])),
                "rerank_count": len(stage.get("post_rerank", [])),
                "baseline_branch_counts": {
                    key: len(rows)
                    for key, rows in stage.get("baseline_branches", {}).items()
                },
                "graph": {
                    "reasons": _safe_graph_reasons(graph.get("reasons")),
                    "ready": graph.get("ready"),
                    "status": graph.get("status"),
                    "branch_statuses": graph.get("branch_statuses"),
                    "candidate_provenance": graph.get("candidate_provenance"),
                    "unique_count": len(graph["candidates"])
                    if graph.get("candidates") is not None
                    else None,
                    "readiness_failure_count": stage.get("readiness_failure_count"),
                },
            }
        )
    for target in targets:
        identities = set(target["chunk_ids"])

        def ranks(key):
            return [
                row.get("rank", rank)
                for stage in stages
                for rank, row in enumerate(stage.get(key, []), 1)
                if _identity(row) in identities
            ]

        packet_ranks = [
            rank for rank, row in enumerate(packet, 1) if _identity(row) in identities
        ]
        coverage = (
            None
            if target.get("answer") is None
            else any(
                target["answer"] in _text(row)
                for row in packet
                if _identity(row) in identities
            )
        )
        result["targets"].append(
            {
                "source_sha256": target["source_sha256"],
                "pool_ranks": ranks("materialized_union"),
                "rerank_ranks": ranks("post_rerank"),
                "packet_ranks": packet_ranks,
                "answer_span_covered": coverage,
            }
        )
    stamps = {}
    for name, data in events:
        if name == "ledger_start":
            result["budget"]["limits"] = data.get("limits")
        if name == "ledger":
            state = data.get("after", {})
            result["budget"]["observed"] = {
                key: state[key]
                for key in (
                    "actions",
                    "sources",
                    "pairs",
                    "text",
                    "planner_calls",
                    "inflight",
                    "closed",
                    "terminal",
                    "remaining_ms",
                )
                if key in state
            }
        if name == "turn_complete":
            result["budget"]["completion"] = {
                key: data[key]
                for key in (
                    "pairs",
                    "actions",
                    "planner_calls",
                    "text",
                    "stop_reason",
                )
                if key in data
            }
        if name == "query_proposals":
            result["proposals"] = {
                "query_count": len(data.get("query_digests", ())),
                "vocabulary_observed": data.get("vocabulary_digest") is not None,
            }
        if "at" in data:
            stamps.setdefault(name, []).append(data["at"])
        if name == "rag_prepared_queries":
            result["prepared"] = {
                "query_count": len(data.get("queries", [])),
                "requested_top_k": data.get("requested_top_k"),
                "candidate_top_k": data.get("candidate_top_k"),
            }
        if name == "rag_search_outcomes":
            result["search_statuses"] = [
                row["status"] for row in data.get("queries", [])
            ]
        if name == "rag_final_selection":
            result["packet_limits"] = data.get("limits")
    for label, first, last in (
        ("retrieval", "rag_prepared_queries", "rag_search_outcomes"),
        ("selection", "rag_search_outcomes", "rag_final_selection"),
    ):
        result["timings_ms"][label] = (
            round((max(stamps[last]) - min(stamps[first])) * 1000, 3)
            if first in stamps and last in stamps
            else None
        )
    return result
