"""Resolved comparison controls and observed operational scenario coverage."""

import os

from .evidence_quality_eval import digest, identity


def comparison_controls():
    from apps.chat.consumers.utils import TOOL_CHUNK_CHAR_LIMIT
    from apps.chat.services.rag_config import (
        direct_rag_candidate_top_k,
        direct_rag_max_queries,
        direct_rag_top_k,
        evidence_token_budget,
        max_snippets_per_doc,
        synthesis_max_tokens,
    )

    return {
        "final_passages": direct_rag_top_k(),
        "candidates_per_action": direct_rag_candidate_top_k(),
        "max_actions": direct_rag_max_queries(),
        "evidence_tokens": evidence_token_budget(),
        "output_tokens": synthesis_max_tokens(),
        "legacy_document_cap": max_snippets_per_doc(),
        "preview_characters": TOOL_CHUNK_CHAR_LIMIT,
        "environment": {
            name: os.getenv(name)
            for name in (
                "OPENAI_CONTEXT_LIMIT",
                "OPENAI_REQUEST_TIMEOUT_SECONDS",
                "OPENAI_TIMEOUT_RETRIES",
                "LLM_STREAM_RESPONSES",
                "TOOL_CALL_TIMEOUT_SECONDS",
                "RAG_CACHE_ENABLED",
                "APP_RERANK_MODEL",
                "APP_RERANK_MODEL_REVISION",
                "APP_RERANK_TOKENIZER",
                "APP_RERANK_TOKENIZER_REVISION",
                "APP_RERANK_CODE_REVISION",
                "APP_RERANK_PAIR_TOKEN_LIMIT",
            )
        },
    }


def scenario_observation(case, trace, result):
    retrievals = [e for e in trace.events if e["event"] == "retrieval_sources"]
    source_rounds = []
    for event in retrievals:
        source_rounds.append(
            [
                identity(trace.sources[(r["doc_id"], r["chunk_id"])])
                for r in event["rows"]
                if (r["doc_id"], r["chunk_id"]) in trace.sources
            ]
        )
    required = {
        g["source_id"]
        for g in case["gold_support"]
        if g["support_id"]
        in {s for c in case["required_claims"] for s in c["support_ids"]}
    }
    reached = None
    if case["scenario"] == "second_search":
        reached = bool(
            len(source_rounds) >= 2
            and not required <= {s[0] for s in source_rounds[0]}
            and required <= {s[0] for rnd in source_rounds[1:] for s in rnd}
        )
    elif case["scenario"] == "global_limit":
        reached = result.get("stop_reason") in (
            "deadline",
            "pairs",
            "actions",
            "optional_reserve",
        )
    elif case["scenario"] == "cancellation":
        reached = result.get("closed") is True
    return {
        "scenario_exercised": reached,
        "acquisition_source_rounds": source_rounds,
        "comparison_controls": comparison_controls(),
    }


def snapshot(case, manifest):
    runtime = manifest.get("runtime", {})
    return {
        "source": digest(case["sources"]),
        "answer": runtime.get("answer_digest"),
        "reranker": runtime.get("reranker_digest"),
        "hardware": runtime.get("hardware_digest"),
        "embedding": runtime.get("embedding_digest"),
        "configuration": digest(comparison_controls()),
        "bindings": digest(manifest["cases"][case["case_id"]]),
    }
