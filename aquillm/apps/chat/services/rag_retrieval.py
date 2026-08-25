"""Deterministic fusion for independently reranked direct-RAG searches."""
from __future__ import annotations

import re
from collections import defaultdict
from math import isfinite
from typing import Any

_RRF_K = 60
_GRAPH_STATUS_PRIORITY = {"miss": 1, "error": 2, "timeout": 3, "hit": 4}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _aggregate_graph_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    statuses = [
        value
        for item in diagnostics
        if (value := item.get("graph_status")) in _GRAPH_STATUS_PRIORITY
    ]
    if statuses:
        safe["graph_status"] = max(
            statuses,
            key=lambda value: _GRAPH_STATUS_PRIORITY[value],
        )

    graph_ms = [
        float(value)
        for item in diagnostics
        if type(value := item.get("graph_ms")) in (int, float)
        and isfinite(float(value))
        and value >= 0
    ]
    if graph_ms:
        safe["graph_ms"] = sum(graph_ms)

    for key, maximum in (
        ("graph_seed_count", 64),
        ("graph_candidate_count", 20),
    ):
        values = [
            value
            for item in diagnostics
            if type(value := item.get(key)) is int and value >= 0
        ]
        if values:
            safe[key] = min(maximum, sum(values))

    for key in ("graph_algorithm_signature", "graph_version_signature"):
        for item in diagnostics:
            value = item.get(key)
            if isinstance(value, str) and _SHA256_RE.fullmatch(value):
                safe[key] = value
                break
    return safe


def _row_identity(row: dict[str, Any]) -> str:
    citation = row.get("citation") or row.get("ref")
    if citation:
        return f"citation:{citation}"
    chunk_id = row.get("chunk_id") or row.get("i")
    doc_id = row.get("doc_id") or row.get("d")
    if chunk_id is None:
        return ""
    return f"chunk:{doc_id}:{chunk_id}"


def merge_ranked_tool_results(
    results: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    """Reciprocal-rank-fuse tool payloads while preserving real cited rows."""
    cap = max(1, int(limit))
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    rows_by_identity: dict[str, dict[str, Any]] = {}
    documents: set[str] = set()
    image_instruction: str | None = None
    private_diagnostics: list[dict[str, Any]] = []
    sequence = 0

    for payload in results:
        if not isinstance(payload, dict):
            continue
        diagnostics = payload.get("_retrieval_diagnostics")
        if isinstance(diagnostics, dict):
            private_diagnostics.append(dict(diagnostics))
        for title in payload.get("retrieved_documents") or []:
            if title:
                documents.add(str(title))
        if image_instruction is None and payload.get("_image_instruction"):
            image_instruction = str(payload["_image_instruction"])
        for fallback_rank, row in enumerate(payload.get("result") or [], start=1):
            if not isinstance(row, dict):
                continue
            identity = _row_identity(row)
            if not identity:
                continue
            raw_rank = row.get("rank") or row.get("r") or fallback_rank
            try:
                rank = max(1, int(raw_rank))
            except (TypeError, ValueError):
                rank = fallback_rank
            scores[identity] += 1.0 / (_RRF_K + rank)
            if identity not in first_seen:
                first_seen[identity] = sequence
                rows_by_identity[identity] = dict(row)
                sequence += 1

    ordered_identities = sorted(
        rows_by_identity,
        key=lambda identity: (-scores[identity], first_seen[identity]),
    )[:cap]
    merged_rows: list[dict[str, Any]] = []
    for merged_rank, identity in enumerate(ordered_identities, start=1):
        row = dict(rows_by_identity[identity])
        if "r" in row and "rank" not in row:
            row["r"] = merged_rank
        else:
            row["rank"] = merged_rank
        merged_rows.append(row)

    if not merged_rows:
        for payload in results:
            if isinstance(payload, dict) and payload.get("retrieval_status") == "no_results":
                return dict(payload)
        return {"result": [], "retrieval_status": "no_results", "retrieved_count": 0}

    merged: dict[str, Any] = {
        "result": merged_rows,
        "retrieval_status": "results_found",
        "retrieved_count": len(merged_rows),
    }
    if documents:
        merged["retrieved_documents"] = sorted(documents)
    if image_instruction:
        merged["_image_instruction"] = image_instruction
    if private_diagnostics:
        safe_graph_diagnostics = _aggregate_graph_diagnostics(private_diagnostics)
        if safe_graph_diagnostics:
            merged["_retrieval_diagnostics"] = safe_graph_diagnostics
    return merged


__all__ = ["merge_ranked_tool_results"]
