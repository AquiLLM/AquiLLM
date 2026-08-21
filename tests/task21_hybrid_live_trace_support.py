import hashlib

ARMS = (
    "vector_only",
    "direct",
    "extended",
    "combined",
    "combined_reranked",
)


def _candidate(arm):
    direct = arm in {"direct", "combined", "combined_reranked"}
    extended = arm in {"extended", "combined", "combined_reranked"}
    sources = ["baseline"]
    if direct:
        sources.append("direct")
    if extended:
        sources.append("extended")
    return {
        "chunk_id": "public-token-001",
        "ordinal": 1,
        "sources": sources,
        "baseline_rank": 1,
        "direct_rank": 1 if direct else None,
        "direct_score_hex": 0.5.hex() if direct else None,
        "extended_rank": 1 if extended else None,
        "extended_score_hex": 0.25.hex() if extended else None,
        "fusion_score_hex": 0.75.hex(),
        "reranker_rank": 1 if arm == "combined_reranked" else None,
    }


def valid_trace():
    return {
        "schema": "task21-hybrid-live-trace-v1",
        "run_id": "a" * 32,
        "source_commit": "b" * 40,
        "fixture_id": "kg-task20-synthetic-v1",
        "fixture_checksum": "c" * 64,
        "manifest_checksum": "d" * 64,
        "arms": {
            arm: [
                {
                    "case_id": "inaccessible_collection_is_excluded",
                    "candidate_trace": [_candidate(arm)],
                    "timing_trace": {
                        "candidate_ms": 0.2,
                        "branch_ms": 0.2,
                        "fusion_ms": 0.2,
                        "rerank_ms": 0.2,
                        "total_ms": 1.0,
                    },
                    "authorization_status": "current",
                    "graph_scheduled": arm != "vector_only",
                    "inaccessible_candidate_count": 0,
                }
            ]
            for arm in ARMS
        },
        "freshness_attestation": {
            "projection_keys": ["e" * 64],
            "generation_keys": ["f" * 64],
            "graph_checksums": ["1" * 64],
            "ready_bundle_checksums": ["2" * 64],
            "ontology_version": "ontology-v1",
            "ontology_checksum": "3" * 64,
        },
        "backend_parity_inputs": [
            {
                "branch": branch,
                "ready_bundle_checksum": "4" * 64,
                "seed_checksum": "5" * 64,
                "seed_count": 1,
                "max_depth": depth,
                "max_nodes": 20,
                "max_edges": 40,
                "max_results": 10,
                "projection_keys": ["6" * 64],
                "generation_keys": ["7" * 64],
                "authorized_document_keys": [character * 64],
            }
            for branch, depth, character in (
                ("direct", 1, "8"),
                ("extended", 2, "9"),
            )
        ],
    }


def valid_observation_attestation(
    trace_body,
    *,
    run_id,
    source_commit,
    config_sha256,
    images,
    projections,
):
    return {
        "schema": "task21-hybrid-live-observation-v1",
        "run_id": run_id,
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "images": images,
        "projection_checksums": projections,
        "artifact_sha256": {
            "observations": "1" * 64,
            "freshness": "2" * 64,
            "backend_parity": "3" * 64,
            "live_trace": hashlib.sha256(trace_body).hexdigest(),
        },
    }
