"""Deterministic fusion for independently reranked direct-RAG searches."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any

from apps.chat.services.rag_config import max_snippets_per_doc
from apps.chat.services.rag_evidence import diversify_evidence_chunks
from apps.documents.services.chunk_rerank_results import RerankScoreSet
from apps.documents.services.chunk_rerank_score_transport import deserialize_score_set

_RRF_K = 60
_GRAPH_STATUS_PRIORITY = {"miss": 1, "error": 2, "timeout": 3, "hit": 4}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CITATION_RE = re.compile(r"\[doc:([^\s\]]+) chunk:([1-9][0-9]*)\]")
_PUBLIC_ROW_KEYS = frozenset({
    "rank", "chunk_id", "doc_id", "chunk", "title", "citation", "text",
    "type", "image_url", "r", "i", "d", "c", "n", "ref", "x", "ty", "u",
})


@dataclass(frozen=True)
class FusedRetrievalPool:
    rows: tuple[dict[str, Any], ...]
    source_score_sets: tuple[RerankScoreSet, ...]
    fused_scores: tuple[tuple[str, float], ...]


def _verified_row_coordinates(row: dict[str, Any]) -> tuple[int, str, int, str] | None:
    """Accept only public rows whose citation agrees with their coordinates."""
    chunk_id = row.get("chunk_id", row.get("i"))
    doc_id = row.get("doc_id", row.get("d"))
    number = row.get("chunk", row.get("c"))
    citation = row.get("citation", row.get("ref"))
    if (
        type(chunk_id) is not int or chunk_id <= 0
        or type(doc_id) is not str or not doc_id
        or type(number) is not int or number < 0
        or type(citation) is not str
    ):
        return None
    match = _CITATION_RE.fullmatch(citation)
    if match is None or match.group(1) != doc_id or int(match.group(2)) != chunk_id:
        return None
    if ("chunk_id" in row and "i" in row and row["chunk_id"] != row["i"]) or (
        "doc_id" in row and "d" in row and row["doc_id"] != row["d"]
    ) or ("chunk" in row and "c" in row and row["chunk"] != row["c"]):
        return None
    return chunk_id, doc_id, number, citation


def _score_set_matches_rows(
    score_set: RerankScoreSet, coordinates: dict[int, tuple[str, int]]
) -> bool:
    if any(pk not in coordinates for pk in score_set.candidate_order):
        return False
    return all(
        score.chunk_pk in coordinates
        and (str(score.document_id), score.chunk_number) == coordinates[score.chunk_pk]
        for score in score_set.scores
    )


def fuse_ranked_tool_results(
    results: list[dict[str, Any]], *, candidate_limit: int = 45
) -> FusedRetrievalPool:
    """Fuse the complete verified union; leave evidence selection to the caller."""
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 45:
        raise ValueError("candidate_limit must be between 1 and 45")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    rows_by_citation: dict[str, dict[str, Any]] = {}
    coordinates_by_pk: dict[int, tuple[str, int]] = {}
    source_score_sets: list[RerankScoreSet] = []
    for payload in results:
        if not isinstance(payload, dict):
            continue
        payload_coordinates: dict[int, tuple[str, int]] = {}
        for fallback_rank, row in enumerate(payload.get("result") or (), start=1):
            if not isinstance(row, dict):
                continue
            verified = _verified_row_coordinates(row)
            if verified is None:
                continue
            pk, doc_id, number, citation = verified
            coordinates = doc_id, number
            if pk in coordinates_by_pk and coordinates_by_pk[pk] != coordinates:
                continue
            if citation in rows_by_citation:
                prior = _verified_row_coordinates(rows_by_citation[citation])
                if prior is None or prior[:3] != verified[:3]:
                    continue
            raw_rank = row.get("rank", row.get("r", fallback_rank))
            rank = raw_rank if type(raw_rank) is int and raw_rank > 0 else fallback_rank
            scores[citation] += 1.0 / (_RRF_K + rank)
            payload_coordinates[pk] = coordinates
            coordinates_by_pk[pk] = coordinates
            if citation not in first_seen:
                first_seen[citation] = len(first_seen)
                rows_by_citation[citation] = {
                    key: value for key, value in row.items() if key in _PUBLIC_ROW_KEYS
                }
        score_set = deserialize_score_set(payload.get("_retrieval_scores"))
        if score_set is not None and _score_set_matches_rows(
            score_set, payload_coordinates
        ):
            source_score_sets.append(score_set)
    ordered = sorted(
        rows_by_citation,
        key=lambda citation: (-scores[citation], first_seen[citation]),
    )[:candidate_limit]
    return FusedRetrievalPool(
        tuple(rows_by_citation[citation] for citation in ordered),
        tuple(source_score_sets),
        tuple((citation, scores[citation]) for citation in ordered),
    )


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
    image_instruction: str | None = None
    private_diagnostics: list[dict[str, Any]] = []
    sequence = 0

    for payload in results:
        if not isinstance(payload, dict):
            continue
        diagnostics = payload.get("_retrieval_diagnostics")
        if isinstance(diagnostics, dict):
            private_diagnostics.append(dict(diagnostics))
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
    )
    # Balance the complete candidate union before the final cutoff; a later
    # packet builder cannot recover another paper once its rows are discarded.
    selected_rows = diversify_evidence_chunks(
        [rows_by_identity[identity] for identity in ordered_identities],
        max_snippets_per_doc(),
    )[:cap]
    merged_rows: list[dict[str, Any]] = []
    for merged_rank, selected_row in enumerate(selected_rows, start=1):
        row = dict(selected_row)
        if "r" in row and "rank" not in row:
            row["r"] = merged_rank
        else:
            row["rank"] = merged_rank
        merged_rows.append(row)

    if not merged_rows:
        for payload in results:
            if (isinstance(payload, dict)
                    and payload.get("retrieval_status") == "no_results"):
                return dict(payload)
        return {"result": [], "retrieval_status": "no_results", "retrieved_count": 0}

    merged: dict[str, Any] = {
        "result": merged_rows,
        "retrieval_status": "results_found",
        "retrieved_count": len(merged_rows),
    }
    documents = {
        str(row.get("title") or row.get("n")) for row in merged_rows
        if row.get("title") or row.get("n")
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


__all__ = ["FusedRetrievalPool", "fuse_ranked_tool_results", "merge_ranked_tool_results"]
