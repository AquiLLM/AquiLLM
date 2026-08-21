"""Field order and bounded scalar shapes for Task21 live traces."""

from __future__ import annotations

import re

TASK21_HYBRID_ARMS = (
    "vector_only",
    "direct",
    "extended",
    "combined",
    "combined_reranked",
)
SCHEMA = "task21-hybrid-live-trace-v1"
FIXTURE_ID = "kg-task20-synthetic-v1"
TOP_FIELDS = (
    "schema",
    "run_id",
    "source_commit",
    "fixture_id",
    "fixture_checksum",
    "manifest_checksum",
    "arms",
    "freshness_attestation",
    "backend_parity_inputs",
)
CASE_FIELDS = (
    "case_id",
    "candidate_trace",
    "timing_trace",
    "authorization_status",
    "graph_scheduled",
    "inaccessible_candidate_count",
)
CANDIDATE_FIELDS = (
    "chunk_id",
    "ordinal",
    "sources",
    "baseline_rank",
    "direct_rank",
    "direct_score_hex",
    "extended_rank",
    "extended_score_hex",
    "fusion_score_hex",
    "reranker_rank",
)
TIMING_FIELDS = (
    "candidate_ms",
    "branch_ms",
    "fusion_ms",
    "rerank_ms",
    "total_ms",
)
FRESHNESS_FIELDS = (
    "projection_keys",
    "generation_keys",
    "graph_checksums",
    "ready_bundle_checksums",
    "ontology_version",
    "ontology_checksum",
)
PARITY_FIELDS = (
    "branch",
    "ready_bundle_checksum",
    "seed_checksum",
    "seed_count",
    "max_depth",
    "max_nodes",
    "max_edges",
    "max_results",
    "projection_keys",
    "generation_keys",
    "authorized_document_keys",
)
SOURCES = ("baseline", "direct", "extended")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
