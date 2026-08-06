"""Deterministic, network-denied offline component evaluation and artifacts."""

from __future__ import annotations

import asyncio
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import signal
import site
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from apps.chat.consumers.chat_receive import _configure_append_tools
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.services.rag_query import build_retrieval_query
from aquillm.memory import promote_profile_facts_for_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage
from lib.llm.types.tools import ToolChoice
from lib.memory import clean_stable_facts, heuristic_facts_from_turn

from .metrics import (
    aggregate_evidence,
    binary_metrics,
    categorical_conformance,
    compare_policies,
    memory_stratum_errors,
    query_conformance,
    score_evidence_case,
)
from .network import deny_network
from .policies import sequential_select
from .schema import (
    canonical_json_bytes,
    load_dataset,
    sha256_file,
    validate_test_manifest,
)

SCHEMA_VERSION = "1.0"
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
CANONICAL_ENV = {
    "RAG_DIRECT_ENABLED": "1",
    "RAG_DIRECT_TOP_K": "10",
    "RAG_QUERY_REWRITE_ENABLED": "0",
    "RAG_EVIDENCE_TOKEN_BUDGET": "3500",
    "RAG_MAX_SNIPPETS_PER_DOC": "3",
    "RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED": "1",
    "TOOL_SEARCH_COMPACT_PAYLOAD": "0",
}
REQUIRED_ARTIFACTS = {
    "manifest.json",
    "routing.jsonl",
    "evidence.jsonl",
    "memory.jsonl",
    "timings.jsonl",
    "tests.json",
    "aggregate.json",
    "routing.csv",
    "evidence.csv",
    "memory.csv",
    "report.md",
    "paper-table.md",
    "COMPLETE",
}
_JSONL_MODULES = ("routing", "evidence", "memory", "timings")
_CLASSIFIER_FIELDS = (
    "requires_rag",
    "wants_figures",
    "wants_whole_document",
    "is_retry",
    "requires_local_tools",
)
_REASONS = (
    "retry_request",
    "local_tool_request",
    "figure_request",
    "explicit_search",
    "collection_backed_question",
    "no_retrieval_needed",
)
_ACTIONS = (
    "retrieve",
    "prompt_select_collection",
    "skip_normal_tool_loop",
    "local_tool_handling",
)
_EXCLUDED_CLAIMS = [
    "No generated-answer correctness, relevance, faithfulness, or "
    "citation-entailment claim.",
    "No end-to-end latency, concurrency, GPU, or production-throughput claim.",
    "No authorization or database-isolation claim from syntax and prefix checks.",
    "No population estimate or sampling-based confidence interval.",
]


@contextmanager
def _canonical_environment():
    previous = {name: os.environ.get(name) for name in CANONICAL_ENV}
    os.environ.update(CANONICAL_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _conversation_for_query(case: dict) -> SimpleNamespace:
    inputs = case["input"]
    messages = []
    for query in inputs.get("prior_vector_queries", []):
        messages.append(
            SimpleNamespace(
                tool_call_name="vector_search",
                tool_call_input={"search_string": query},
                tool_name=None,
                result_dict=None,
            )
        )
    for title in inputs.get("prior_document_titles", []):
        messages.append(
            SimpleNamespace(
                tool_call_name=None,
                tool_call_input=None,
                tool_name="vector_search",
                result_dict={"retrieved_documents": [title]},
            )
        )
    return SimpleNamespace(messages=messages)


def _helper_action(case: dict) -> str:
    inputs = case["input"]
    document_tool = object()
    all_tool = object()
    prior = [document_tool] if inputs.get("prior_tools") else None
    tools, _choice = _configure_append_tools(
        message_content=inputs["text"],
        all_tools=[all_tool],
        document_tools=[document_tool],
        selected_collection_ids=inputs.get("selected_collection_ids", []),
        prior_user_tools=prior,
        prior_user_tool_choice=ToolChoice(type="any") if prior else None,
    )
    if tools and tools[0] is all_tool:
        return "local_tool_handling"
    if tools:
        return (
            "retrieve"
            if inputs.get("selected_collection_ids")
            else "prompt_select_collection"
        )
    return "skip_normal_tool_loop"


def _direct_action(case: dict, event_loop: asyncio.AbstractEventLoop) -> str:
    from apps.chat.services import rag_pipeline

    inputs = case["input"]
    convo = Conversation(
        system="offline", messages=[UserMessage(content=inputs["text"])]
    )
    consumer = SimpleNamespace(
        col_ref=SimpleNamespace(collections=inputs.get("selected_collection_ids", [])),
        user=SimpleNamespace(id=1),
        convo=convo,
    )
    retrieved = []

    def fake_search(_consumer, query, top_k):
        retrieved.append((query, top_k))
        return {
            "result": [
                {
                    "evidence_id": "pipeline-synthetic",
                    "doc_id": "pipeline-doc",
                    "chunk_id": 1,
                    "text": "Synthetic pipeline evidence.",
                    "citation": "[doc:pipeline-doc chunk:1]",
                }
            ],
            "retrieved_count": 1,
            "retrieval_status": "results_found",
        }

    async def fake_synthesis(_llm, working_convo, _packet, stream_func=None):
        return working_convo + [
            AssistantMessage(content="Synthetic answer.", stop_reason="end_turn")
        ]

    with (
        patch.object(rag_pipeline, "_run_vector_search", side_effect=fake_search),
        patch.object(
            rag_pipeline, "synthesize_from_evidence", side_effect=fake_synthesis
        ),
        patch.object(rag_pipeline, "log_direct_rag_turn"),
    ):
        outcome = event_loop.run_until_complete(
            run_direct_rag_turn(consumer, object(), convo)
        )
    if outcome == "skipped":
        return "skip_normal_tool_loop"
    return "retrieve" if retrieved else "prompt_select_collection"


def _classifier_dict(intent) -> dict[str, bool]:
    return {field: bool(getattr(intent, field)) for field in _CLASSIFIER_FIELDS}


def _routing_records(
    cases: list[dict], event_loop: asyncio.AbstractEventLoop
) -> list[dict]:
    records = []
    for case in cases:
        inputs = case["input"]
        intent = classify_chat_message(
            inputs["text"],
            selected_collection_ids=inputs.get("selected_collection_ids", []),
            prior_tools=inputs.get("prior_tools", []),
        )
        query = build_retrieval_query(_conversation_for_query(case), inputs["text"])
        helper_action = _helper_action(case)
        direct_action = _direct_action(case, event_loop)
        expected_query = case["gold"].get("expected_query")
        gold = case["gold"]
        if "direct_pipeline_action" in gold:
            expected_direct = gold["direct_pipeline_action"]
        elif (
            not gold["classifier"]["requires_local_tools"]
            and not gold["classifier"]["is_retry"]
            and gold["production_action"]
            in {
                "retrieve",
                "prompt_select_collection",
                "skip_normal_tool_loop",
            }
        ):
            expected_direct = gold["production_action"]
        else:
            expected_direct = None
        actual = {
            "classifier": _classifier_dict(intent),
            "reason": intent.reason,
            "helper_action": helper_action,
            "direct_action": direct_action,
            "query": query,
        }
        expected = {
            "classifier": case["gold"]["classifier"],
            "reason": case["gold"]["reason"],
            "helper_action": case["gold"]["production_action"],
            "direct_action": expected_direct,
            "query": expected_query,
        }
        direct_check = (
            {
                "status": "not_applicable",
                "conformant": None,
                "reason": "fixture has no direct_pipeline_action gold label",
            }
            if expected_direct is None
            else {
                "status": "ok",
                "conformant": direct_action == expected_direct,
            }
        )
        checks = {
            "classifier": actual["classifier"] == expected["classifier"],
            "reason": actual["reason"] == expected["reason"],
            "helper_action": helper_action == expected["helper_action"],
            "direct_action": direct_check,
            "query": expected_query is None or query.strip() == expected_query.strip(),
        }
        conformant = all(
            (
                checks["classifier"],
                checks["reason"],
                checks["helper_action"],
                checks["query"],
                direct_check["status"] == "not_applicable"
                or direct_check["conformant"],
            )
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "module": "routing",
                "case_id": case["id"],
                "stratum": case["stratum"],
                "expected": expected,
                "actual": actual,
                "conformant": conformant,
                "diagnostics": {
                    "checks": checks,
                    "rationale": case.get("rationale", ""),
                },
            }
        )
    return records


def _evidence_records(cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        candidates = []
        for candidate in case["candidates"]:
            item = copy.deepcopy(candidate)
            if item.get("observed_citation") is not None:
                item["citation"] = item["observed_citation"]
            candidates.append(item)
        packet = build_evidence_packet(
            {"result": candidates, "retrieval_status": "results_found"},
            query=case["question"],
            search_scope="synthetic fixtures",
            token_budget=case["token_budget"],
        )
        sequential = sequential_select(candidates, case["token_budget"])
        aquillm_score = score_evidence_case(case, packet.chunks)
        sequential_score = score_evidence_case(case, sequential["chunks"])
        expected_ids = case["gold"]["relevant_evidence_ids"]
        selected_ids = [item["evidence_id"] for item in packet.chunks]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "module": "evidence",
                "case_id": case["id"],
                "stratum": case["stratum"],
                "expected": {"relevant_evidence_ids": expected_ids},
                "actual": {"selected_evidence_ids": selected_ids},
                "conformant": set(expected_ids).issubset(selected_ids),
                "aquillm": aquillm_score,
                "sequential": sequential_score,
                "diagnostics": {
                    "aquillm_packet_citation_tokens": packet.citation_tokens,
                    "sequential_citation_tokens": sequential["citation_tokens"],
                },
            }
        )
    return records


def _memory_records(cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        inputs = case["input"]
        raw = heuristic_facts_from_turn(
            inputs["user_content"], inputs["assistant_content"]
        )
        actual = clean_stable_facts(list(dict.fromkeys(raw)))
        expected = case["gold"]["normalized_facts"]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "module": "memory",
                "case_id": case["id"],
                "stratum": case["stratum"],
                "expected": expected,
                "actual": actual,
                "conformant": set(actual) == set(expected),
                "diagnostics": {"raw_facts": raw},
            }
        )
    return records


def _memory_fallback_reachability(cases: list[dict]) -> dict:
    import aquillm.memory as memory_module

    explicit = next(
        case for case in cases if "remember" in case["input"]["user_content"].lower()
    )
    heuristic = next(
        case
        for case in cases
        if "remember" not in case["input"]["user_content"].lower()
        and heuristic_facts_from_turn(
            case["input"]["user_content"], case["input"]["assistant_content"]
        )
    )
    persisted: list[list[str]] = []
    fake_user = SimpleNamespace(id=7)
    counters = {
        "remote_attempt_count": 0,
        "has_remember_intent_calls": 0,
        "normalize_calls": 0,
        "heuristic_calls": 0,
    }
    original_has_remember = memory_module.has_remember_intent
    original_normalize = memory_module.normalize_remember_fact
    original_heuristic = memory_module.heuristic_facts_from_turn

    def fail_immediately(*_args, **_kwargs):
        counters["remote_attempt_count"] += 1
        raise RuntimeError("controlled offline extraction failure")

    def persist(_user, facts):
        persisted.append(list(facts))

    def observed_has_remember(*args, **kwargs):
        counters["has_remember_intent_calls"] += 1
        return original_has_remember(*args, **kwargs)

    def observed_normalize(*args, **kwargs):
        counters["normalize_calls"] += 1
        return original_normalize(*args, **kwargs)

    def observed_heuristic(*args, **kwargs):
        counters["heuristic_calls"] += 1
        return original_heuristic(*args, **kwargs)

    def execute(case):
        before = len(persisted)
        before_counts = dict(counters)
        start = time.perf_counter()
        count = promote_profile_facts_for_turn(
            7,
            case["input"]["user_content"],
            case["input"]["assistant_content"],
        )
        elapsed = time.perf_counter() - start
        observed = {name: counters[name] - before_counts[name] for name in counters}
        if observed["heuristic_calls"]:
            branch = "heuristic"
        elif observed["normalize_calls"]:
            branch = "explicit_remember"
        else:
            branch = "none"
        return {
            "branch": branch,
            "fact_count": count,
            "facts": persisted[before] if len(persisted) > before else [],
            "latency_seconds": elapsed,
            **observed,
        }

    with (
        patch("aquillm.memory.User.objects.filter") as lookup,
        patch("aquillm.memory._promote_profile_facts", side_effect=persist),
        patch(
            "lib.memory.extraction.stable_facts.requests.post",
            side_effect=fail_immediately,
        ),
        patch.object(
            memory_module, "has_remember_intent", side_effect=observed_has_remember
        ),
        patch.object(
            memory_module, "normalize_remember_fact", side_effect=observed_normalize
        ),
        patch.object(
            memory_module, "heuristic_facts_from_turn", side_effect=observed_heuristic
        ),
    ):
        lookup.return_value.first.return_value = fake_user
        explicit_observed = execute(explicit)
        heuristic_observed = execute(heuristic)
    return {
        "orchestration_failure": "controlled_immediate_extraction_failure",
        "explicit_remember": explicit_observed,
        "heuristic": heuristic_observed,
        "network_failure_latency_seconds": explicit_observed["latency_seconds"]
        + heuristic_observed["latency_seconds"],
    }


def _timing_record(module: str, operation, repeats: int, input_size: dict) -> dict:
    operation()  # warm-up, deliberately excluded from samples and conformance
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    median = statistics.median(samples)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "schema_version": SCHEMA_VERSION,
        "module": module,
        "phase": "timing",
        "warmup_count": 1,
        "repeat_count": repeats,
        "raw_samples_seconds": samples,
        "median_seconds": median,
        "p95_seconds": p95,
        "throughput_per_second": None if median == 0 else 1.0 / median,
        "input_size": input_size,
        "perf_counter_resolution_seconds": time.get_clock_info(
            "perf_counter"
        ).resolution,
    }


def _timings(datasets: dict, repeats: int) -> list[dict]:
    if repeats < 1:
        raise ValueError("timing_repeats must be positive")
    routing_case = datasets["routing"]["cases"][0]
    evidence_case = next(
        case for case in datasets["evidence"]["cases"] if case["candidates"]
    )
    memory_case = datasets["memory"]["cases"][0]
    timings = [
        _timing_record(
            "routing",
            lambda: classify_chat_message(
                routing_case["input"]["text"],
                selected_collection_ids=routing_case["input"][
                    "selected_collection_ids"
                ],
                prior_tools=routing_case["input"].get("prior_tools", []),
            ),
            repeats,
            {"characters": len(routing_case["input"]["text"])},
        ),
        _timing_record(
            "memory",
            lambda: heuristic_facts_from_turn(
                memory_case["input"]["user_content"],
                memory_case["input"]["assistant_content"],
            ),
            repeats,
            {
                "characters": len(memory_case["input"]["user_content"])
                + len(memory_case["input"]["assistant_content"])
            },
        ),
    ]
    for candidate_count in (1, 10, 100):
        candidates = _scaled_evidence_candidates(evidence_case, candidate_count)
        timings.insert(
            -1,
            _timing_record(
                "evidence",
                lambda candidates=candidates: build_evidence_packet(
                    {"result": candidates},
                    query=evidence_case["question"],
                    search_scope="synthetic fixtures",
                    token_budget=1_000_000,
                ),
                repeats,
                {"candidate_count": candidate_count},
            ),
        )
    return timings


def _scaled_evidence_candidates(case: dict, count: int) -> list[dict]:
    base = case["candidates"]
    candidates = []
    for index in range(count):
        item = copy.deepcopy(base[index % len(base)])
        item["evidence_id"] = f"timing-evidence-{index + 1}"
        item["doc_id"] = f"timing-doc-{index % 10 + 1}"
        item["chunk_id"] = index + 1
        item["citation"] = f"[doc:{item['doc_id']} chunk:{item['chunk_id']}]"
        candidates.append(item)
    return candidates


def _ratio(numerator: int | float, denominator: int | float) -> dict:
    return {
        "status": "not_applicable" if denominator == 0 else "ok",
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _by_stratum(
    records: list[dict], scorer, *, strata: list[str] | None = None
) -> dict:
    strata = strata or sorted({record["stratum"] for record in records})
    return {
        stratum: scorer([record for record in records if record["stratum"] == stratum])
        for stratum in strata
    }


def _categorical_summary(records: list[dict], expected_key: str, actual_key: str):
    return categorical_conformance(
        [record["expected"][expected_key] for record in records],
        [record["actual"][actual_key] for record in records],
        labels=list(_ACTIONS if "action" in expected_key else _REASONS),
    )


def _query_summary(records: list[dict]) -> dict:
    return query_conformance(
        [record["expected"]["query"] for record in records],
        [record["actual"]["query"] for record in records],
    )


def _metric_pool(records: list[dict], key: str) -> dict:
    applicable = [record[key] for record in records if record[key]["status"] == "ok"]
    return _ratio(
        sum(metric["numerator"] for metric in applicable),
        sum(metric["denominator"] for metric in applicable),
    )


def _summarize_evidence_policy(records: list[dict]) -> dict:
    recall = aggregate_evidence(records)
    doc_applicable = [
        record["relevant_document_coverage"]
        for record in records
        if record["relevant_document_coverage"]["status"] == "ok"
    ]
    diversity_total = sum(record["distinct_selected_documents"] for record in records)
    token_total = sum(record["estimated_token_use"] for record in records)
    overrun_total = sum(record["overrun_tokens"] for record in records)
    citations = [record["citation_diagnostics"] for record in records]
    return {
        "support": len(records),
        **recall,
        "macro_relevant_document_coverage": _ratio(
            sum(metric["value"] for metric in doc_applicable), len(doc_applicable)
        ),
        "relevant_document_coverage": _metric_pool(
            records, "relevant_document_coverage"
        ),
        "selected_document_diversity": {
            "support": len(records),
            "total": diversity_total,
            "mean": _ratio(diversity_total, len(records)),
        },
        "estimated_token_use": {
            "support": len(records),
            "total": token_total,
            "mean": _ratio(token_total, len(records)),
        },
        "overrun_tokens": {
            "support": len(records),
            "total": overrun_total,
            "mean": _ratio(overrun_total, len(records)),
            "within_budget": _ratio(
                sum(record["overrun_tokens"] == 0 for record in records),
                len(records),
            ),
        },
        "citation_syntax_validity": _metric_pool(
            [{"syntax": diagnostic["syntax_validity"]} for diagnostic in citations],
            "syntax",
        ),
        "citation_chunk_consistency": _metric_pool(
            [
                {"consistency": diagnostic["chunk_consistency"]}
                for diagnostic in citations
            ],
            "consistency",
        ),
        "duplicate_citation_count": sum(
            diagnostic["duplicate_count"] for diagnostic in citations
        ),
        "conflicting_citation_count": sum(
            diagnostic["conflict_count"] for diagnostic in citations
        ),
        "image_path_prefix_behavior": _metric_pool(
            [
                {"image": diagnostic["image_path_prefix_behavior"]}
                for diagnostic in citations
            ],
            "image",
        ),
    }


def _aggregate_evidence_policy(records: list[dict]) -> dict:
    return {
        "overall": _summarize_evidence_policy(records),
        "by_stratum": _by_stratum(records, _summarize_evidence_policy),
    }


def _aggregate(routing: list[dict], evidence: list[dict], memory: list[dict], timings):
    classifier = {}
    for field in _CLASSIFIER_FIELDS:

        def score_field(records, field=field):
            return binary_metrics(
                [record["expected"]["classifier"][field] for record in records],
                [record["actual"]["classifier"][field] for record in records],
            )

        classifier[field] = {
            **score_field(routing),
            "by_stratum": _by_stratum(routing, score_field),
        }

    def score_reason(records):
        return categorical_conformance(
            [record["expected"]["reason"] for record in records],
            [record["actual"]["reason"] for record in records],
            labels=list(_REASONS),
        )

    def score_helper(records):
        return _categorical_summary(records, "helper_action", "helper_action")

    def score_direct(records):
        return _categorical_summary(records, "direct_action", "direct_action")

    reason = {**score_reason(routing), "by_stratum": _by_stratum(routing, score_reason)}
    helper_action = {
        **score_helper(routing),
        "by_stratum": _by_stratum(routing, score_helper),
    }
    direct_records = [
        record for record in routing if record["expected"]["direct_action"] is not None
    ]
    direct_action = {
        **score_direct(direct_records),
        "not_applicable": len(routing) - len(direct_records),
        "by_stratum": _by_stratum(
            direct_records,
            score_direct,
            strata=sorted({record["stratum"] for record in routing}),
        ),
    }
    query_records = [
        record for record in routing if record["expected"]["query"] is not None
    ]
    query = {
        **_query_summary(query_records),
        "not_applicable": len(routing) - len(query_records),
        "by_stratum": _by_stratum(
            query_records,
            _query_summary,
            strata=sorted({record["stratum"] for record in routing}),
        ),
    }
    policy_pairs = []
    for record in evidence:
        pair = {
            "aquillm": dict(record["aquillm"]),
            "sequential": dict(record["sequential"]),
        }
        for policy in ("aquillm", "sequential"):
            pair[policy].update(pair[policy]["citation_diagnostics"])
            pair[policy]["image_path_prefix_behavior"] = pair[policy][
                "citation_diagnostics"
            ]["image_path_prefix_behavior"]
        policy_pairs.append(pair)
    comparison_metrics = (
        "relevant_evidence_recall",
        "relevant_document_coverage",
        "distinct_selected_documents",
        "estimated_token_use",
        "overrun_tokens",
        "syntax_validity",
        "chunk_consistency",
        "duplicate_count",
        "conflict_count",
        "image_path_prefix_behavior",
    )
    timing_aggregate = {}
    for record in timings:
        timing_aggregate.setdefault(record["module"], []).append(record)
    paired_comparisons = {
        metric: compare_policies(policy_pairs, metric) for metric in comparison_metrics
    }
    paired_comparisons["distinct_selected_documents"]["interpretation"] = (
        "descriptive_not_quality"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "routing": {
            "support": len(routing),
            "classifier": classifier,
            "reason": reason,
        },
        "action": {"helper": helper_action, "direct": direct_action},
        "query": query,
        "evidence": {
            "aquillm": _aggregate_evidence_policy(
                [record["aquillm"] for record in evidence]
            ),
            "sequential": _aggregate_evidence_policy(
                [record["sequential"] for record in evidence]
            ),
            "paired_comparisons": paired_comparisons,
        },
        "memory": memory_stratum_errors(memory),
        "tests": {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "unavailable": 0,
        },
        "timing": timing_aggregate,
        "excluded_claims": list(_EXCLUDED_CLAIMS),
    }


def run_component_evaluation(fixture_dir: Path, timing_repeats: int) -> dict:
    """Execute approved fixtures once for conformance under socket denial."""
    fixture_dir = Path(fixture_dir)
    datasets = {
        kind: load_dataset(fixture_dir / f"{kind}.yaml", kind)
        for kind in ("routing", "evidence", "memory")
    }
    event_loop = asyncio.new_event_loop()
    try:
        with _canonical_environment(), deny_network() as attempts:
            routing = _routing_records(datasets["routing"]["cases"], event_loop)
            evidence = _evidence_records(datasets["evidence"]["cases"])
            memory = _memory_records(datasets["memory"]["cases"])
            memory_fallback = _memory_fallback_reachability(datasets["memory"]["cases"])
            timings = _timings(datasets, timing_repeats)
    finally:
        event_loop.close()
    aggregate = _aggregate(routing, evidence, memory, timings)
    aggregate["memory"]["fallback_reachability"] = {
        "orchestration_failure": memory_fallback["orchestration_failure"],
        "explicit_remember": {
            key: value
            for key, value in memory_fallback["explicit_remember"].items()
            if key != "latency_seconds"
        },
        "heuristic": {
            key: value
            for key, value in memory_fallback["heuristic"].items()
            if key != "latency_seconds"
        },
    }
    aggregate["timing"]["memory_fallback"] = {
        "explicit_remember_latency_seconds": memory_fallback["explicit_remember"][
            "latency_seconds"
        ],
        "heuristic_latency_seconds": memory_fallback["heuristic"]["latency_seconds"],
        "network_failure_latency_seconds": memory_fallback[
            "network_failure_latency_seconds"
        ],
    }
    return {
        "routing": routing,
        "evidence": evidence,
        "memory": memory,
        "timings": timings,
        "memory_fallback": memory_fallback,
        "canonical_env": dict(CANONICAL_ENV),
        "network_attempts": {
            "total": attempts.total,
            "details": attempts.details,
            "scope": "component_execution_only",
        },
        "aggregate": aggregate,
    }


def _subprocess_environment(module_root: Path) -> dict[str, str]:
    """Build a minimal environment without ambient credentials or proxies."""
    runtime_names = {
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in runtime_names and value
    }
    python_paths = [str(Path(module_root).resolve())]
    for module_name in ("pytest", "yaml", "django"):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.submodule_search_locations is None:
            raise RuntimeError(
                f"required test dependency is unavailable: {module_name}"
            )
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        python_paths.append(str(package_dir.resolve().parent))
    python_paths.extend(site.getsitepackages())
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        python_paths.append(user_site)
    else:
        python_paths.extend(user_site)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "DJANGO_SETTINGS_MODULE": "aquillm.settings",
            "SECRET_KEY": "offline-test-only",
            "GOOGLE_OAUTH2_CLIENT_ID": "offline-test-only",
            "GOOGLE_OAUTH2_CLIENT_SECRET": "offline-test-only",
            "OPENAI_API_KEY": "offline-test-only",
            "GEMINI_API_KEY": "offline-test-only",
            "ANTHROPIC_API_KEY": "offline-test-only",
        }
    )
    return environment


def _terminate_process_tree(process) -> None:
    """Forcefully terminate a pytest process and all descendants."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _read_network_audit(path: Path) -> tuple[dict, str | None]:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
        total = audit["total"]
        details = audit["details"]
        if not isinstance(total, int) or total < 0 or not isinstance(details, list):
            raise ValueError
        if total != len(details):
            raise ValueError
        return {"status": "available", "total": total, "details": details}, None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return (
            {"status": "unavailable", "total": None, "details": []},
            "subprocess_initialization_or_network_audit_failure",
        )


def run_test_manifest(
    path: Path,
    project_root: Path,
    *,
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
) -> dict:
    """Run every included exact node once and report blocked entries as unavailable."""
    path, project_root = Path(path), Path(project_root)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_test_manifest(data)
    included = [entry for entry in data["entries"] if entry["status"] == "included"]
    for entry in included:
        validate_test_manifest(
            {"schema_version": "1.0", "entries": [entry]},
            project_root=project_root,
        )
    if timeout_seconds <= 0:
        raise ValueError("test timeout must be positive")
    module_root = Path(__file__).resolve().parents[4]
    env = _subprocess_environment(module_root)
    with tempfile.TemporaryDirectory(prefix="aquillm-junit-") as temp:
        junit = Path(temp) / "junit.xml"
        network_audit_path = Path(temp) / "network-attempts.json"
        env["AQUILLM_OFFLINE_NETWORK_ATTEMPTS_FILE"] = str(network_audit_path)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "apps.chat.evals.offline.pytest_no_network",
            *[entry["node_id"] for entry in included],
            "--junitxml",
            str(junit),
            "-q",
        ]
        popen_options = {
            "cwd": project_root,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if os.name == "nt":
            popen_options["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(command, **popen_options)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            exit_code = 124
        counts, node_results = _parse_junit(
            junit, [entry["node_id"] for entry in included]
        )
        network_attempts, audit_failure = _read_network_audit(network_audit_path)
    integrity_failure = None
    if timed_out:
        integrity_failure = "pytest_timeout"
    elif audit_failure:
        integrity_failure = audit_failure
    elif network_attempts["total"]:
        integrity_failure = "subprocess_network_attempt"
    entries = []
    for entry in data["entries"]:
        item = dict(entry)
        if entry["status"] == "included":
            observed = node_results[entry["node_id"]]
            item["outcome"] = "timeout" if timed_out else observed["outcome"]
            item["instances"] = observed["instances"]
        else:
            item["outcome"] = "unavailable"
        entries.append(item)
    counts["unavailable"] = len(data["entries"]) - len(included)
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "summary": counts,
        "exit_code": exit_code,
        "stdout": _redact_subprocess_output(stdout),
        "stderr": _redact_subprocess_output(stderr),
        "command": [
            "python",
            "-m",
            "pytest",
            "<exact-manifest-nodes>",
            "--junitxml",
            "<temporary>",
        ],
        "network_scope": "component_and_pytest_subprocess",
        "declared_network_policy": "no_network",
        "enforced_subprocess_network_denial": True,
        "subprocess_network_attempts": network_attempts,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "integrity_failure": integrity_failure,
    }


def _parse_junit(path: Path, node_ids: list[str]) -> tuple[dict, dict]:
    if not path.is_file():
        return (
            {
                "collected": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": len(node_ids),
            },
            {
                node_id: {
                    "outcome": "missing",
                    "instances": _instance_counts([]),
                }
                for node_id in node_ids
            },
        )
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    passed = collected - failed - errors - skipped
    observed_cases = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        observed_cases.append((case, outcome))
    by_node = {}
    for node_id in node_ids:
        outcomes = [
            outcome
            for case, outcome in observed_cases
            if _junit_case_matches_node(case, node_id)
        ]
        if "error" in outcomes:
            node_outcome = "error"
        elif "failed" in outcomes:
            node_outcome = "failed"
        elif "skipped" in outcomes:
            node_outcome = "skipped"
        elif outcomes:
            node_outcome = "passed"
        else:
            node_outcome = "missing"
        by_node[node_id] = {
            "outcome": node_outcome,
            "instances": _instance_counts(outcomes),
        }
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }, by_node


def _instance_counts(outcomes: list[str]) -> dict:
    return {
        "collected": len(outcomes),
        "passed": outcomes.count("passed"),
        "failed": outcomes.count("failed"),
        "skipped": outcomes.count("skipped"),
        "errors": outcomes.count("error"),
    }


def _junit_case_matches_node(case: ET.Element, node_id: str) -> bool:
    file_selector, *selectors = node_id.replace("\\", "/").split("::")
    case_name = case.attrib.get("name", "").split("[", 1)[0]
    if not selectors or case_name != selectors[-1]:
        return False
    case_file = case.attrib.get("file", "").replace("\\", "/")
    classname = case.attrib.get("classname", "")
    if len(selectors) == 2 and classname.split(".")[-1] != selectors[0]:
        return False
    if case_file:
        return case_file.endswith(file_selector)
    module_parts = classname.split(".")
    if len(selectors) == 2:
        module_parts = module_parts[:-1]
    module_path = "/".join(module_parts) + ".py"
    return module_path.endswith(file_selector)


def _redact_subprocess_output(raw: str) -> str:
    """Retain diagnostic text without machine paths, credentials, or timings."""
    text = raw or ""
    text = re.sub(r"[A-Za-z]:\\[^\r\n:]+", "<path>", text)
    text = re.sub(r"/(?:home|Users|tmp)/[^\s:]+", "<path>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?s\b", "<elapsed>", text)
    text = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", "<timestamp>", text
    )
    for private in (
        "offline-test-only",
        os.getenv("USERNAME", ""),
        platform.node(),
    ):
        if private:
            text = re.sub(re.escape(private), "<redacted>", text, flags=re.IGNORECASE)
    return text


def _jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def render_csv(records: list[dict]) -> str:
    """Render evaluation records as deterministic CSV without filesystem access."""
    fields = (
        "schema_version",
        "module",
        "case_id",
        "stratum",
        "conformant",
        "expected",
        "actual",
        "diagnostics",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                field: json.dumps(
                    record.get(field),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if field in {"expected", "actual", "diagnostics"}
                else record.get(field)
                for field in fields
            }
        )
    return handle.getvalue()


def regenerate_paper_table(aggregate_path: Path) -> str:
    """Render the paper table using aggregate JSON as the sole data source."""
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    return _paper_table_text(aggregate)


def _paper_table_text(aggregate: dict) -> str:

    def ratio(value):
        if not isinstance(value, dict) or value.get("status") != "ok":
            return "N/A"
        return f"{value['numerator']}/{value['denominator']} ({value['value']:.3f})"

    def nested(*keys):
        value = aggregate
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    tests_result = (
        f"{aggregate['tests'].get('passed', 0)}/"
        f"{aggregate['tests'].get('collected', 0)}"
    )
    rows = [
        (
            "Routing reason conformance",
            ratio(
                nested("routing", "reason", "conformance")
                or nested("routing", "conformance")
            ),
        ),
        ("Helper action conformance", ratio(nested("action", "helper", "conformance"))),
        ("Direct action conformance", ratio(nested("action", "direct", "conformance"))),
        ("Query conformance", ratio(nested("query", "conformance"))),
        (
            "AquiLLM evidence macro recall",
            ratio(
                nested(
                    "evidence",
                    "aquillm",
                    "overall",
                    "macro_relevant_evidence_recall",
                )
                or nested("evidence", "aquillm", "macro_relevant_evidence_recall")
            ),
        ),
        (
            "Memory exact-set conformance",
            ratio(nested("memory", "overall", "exact_set_conformance")),
        ),
        (
            "Included contract tests passed",
            tests_result,
        ),
    ]
    lines = [
        "# Preliminary offline component evaluation",
        "",
        "| Measure | Result |",
        "|---|---:|",
        *[f"| {label} | {value} |" for label, value in rows],
        "",
        "Limitations: " + " ".join(aggregate.get("excluded_claims", [])),
        "",
    ]
    return "\n".join(lines)


def render_report(aggregate: dict) -> str:
    """Render the human-readable report solely from aggregate JSON data."""
    tests = aggregate["tests"]
    test_lines = "".join(
        f"- {name}: {tests.get(name, 0)}\n"
        for name in (
            "collected",
            "passed",
            "failed",
            "skipped",
            "errors",
            "unavailable",
        )
    )
    timing_lines = [
        "| Module | Input size | Median seconds | p95 seconds | Throughput/s |",
        "|---|---|---:|---:|---:|",
    ]
    for module, value in aggregate.get("timing", {}).items():
        records = value if isinstance(value, list) else [value]
        for record in records:
            if not isinstance(record, dict) or "median_seconds" not in record:
                continue
            timing_lines.append(
                "| "
                + " | ".join(
                    (
                        module,
                        json.dumps(record.get("input_size", {}), sort_keys=True),
                        f"{record.get('median_seconds', 0):.9f}",
                        f"{record.get('p95_seconds', 0):.9f}",
                        f"{record.get('throughput_per_second', 0):.3f}",
                    )
                )
                + " |"
            )
    detailed_metrics = {
        name: aggregate[name]
        for name in ("routing", "action", "query", "evidence", "memory")
    }
    return (
        "# Preliminary offline component evaluation report\n\n"
        "This report separates deterministic component conformance, contract-test "
        "counts, and local microbenchmarks. Fixed-set misses remain visible and are "
        "not integrity failures.\n\n"
        + _paper_table_text(aggregate)
        + "\n## Contract tests\n\n"
        + test_lines
        + "\n## Timings\n\n"
        + "\n".join(timing_lines)
        + "\n\n"
        + "## Detailed aggregate metrics\n\n```json\n"
        + json.dumps(detailed_metrics, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\n"
        "## Excluded claims\n\n"
        + "".join(f"- {claim}\n" for claim in aggregate["excluded_claims"])
    )


def _report_text(aggregate: dict) -> str:
    """Backward-compatible internal alias for the pure report renderer."""
    return render_report(aggregate)


def write_artifacts(result: dict, output_dir: Path) -> None:
    """Atomically create one immutable, complete artifact directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"artifact output already exists: {output_dir}")
    payload = copy.deepcopy(result)
    manifest = payload["manifest"]
    manifest.setdefault("fixture_sensitivity", "synthetic_public")
    _validate_result_contract(payload)
    _scan_sensitive(payload)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        (temp_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        for module in _JSONL_MODULES:
            (temp_dir / f"{module}.jsonl").write_bytes(_jsonl_bytes(payload[module]))
        (temp_dir / "tests.json").write_bytes(canonical_json_bytes(payload["tests"]))
        (temp_dir / "aggregate.json").write_bytes(
            canonical_json_bytes(payload["aggregate"])
        )
        for module in ("routing", "evidence", "memory"):
            (temp_dir / f"{module}.csv").write_text(
                render_csv(payload[module]), encoding="utf-8", newline=""
            )
        (temp_dir / "report.md").write_text(
            render_report(payload["aggregate"]), encoding="utf-8", newline="\n"
        )
        (temp_dir / "paper-table.md").write_text(
            regenerate_paper_table(temp_dir / "aggregate.json"),
            encoding="utf-8",
            newline="\n",
        )
        hashes = {
            name: sha256_file(temp_dir / name)
            for name in sorted(REQUIRED_ARTIFACTS - {"COMPLETE"})
        }
        (temp_dir / "COMPLETE").write_bytes(
            canonical_json_bytes({"schema_version": SCHEMA_VERSION, "sha256": hashes})
        )
        validate_artifacts(temp_dir)
        os.replace(temp_dir, output_dir)
    except Exception:
        for path in sorted(temp_dir.glob("*"), reverse=True):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise


def _validate_result_contract(result: dict) -> None:
    required_manifest = {
        "schema_version",
        "run_id",
        "timestamp_utc",
        "source_commit",
        "source_dirty",
        "fixture_hashes",
        "code_hashes",
        "config_hashes",
        "canonical_env",
        "environment",
        "component_network_attempts",
        "test_manifest_hash",
    }
    missing = required_manifest - set(result.get("manifest", {}))
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")
    if result["manifest"]["source_dirty"] is not False:
        raise ValueError("canonical source must be clean")
    if result["manifest"].get("fixture_sensitivity") != "synthetic_public":
        raise ValueError("fixture sensitivity must be synthetic_public")
    required_aggregate = {
        "routing",
        "action",
        "query",
        "evidence",
        "memory",
        "tests",
        "timing",
        "excluded_claims",
    }
    missing = required_aggregate - set(result.get("aggregate", {}))
    if missing:
        raise ValueError(f"aggregate missing required fields: {sorted(missing)}")
    for module in ("routing", "evidence", "memory"):
        for record in result[module]:
            required = {
                "schema_version",
                "module",
                "case_id",
                "stratum",
                "expected",
                "actual",
                "conformant",
                "diagnostics",
            }
            if required - set(record):
                raise ValueError(f"{module} record missing required fields")
    if not isinstance(result["tests"].get("entries"), list):
        raise ValueError("tests.json must represent every manifest entry")


def validate_artifacts(output_dir: Path) -> None:
    """Validate exact membership, hashes, schema, table rendering, and safety."""
    output_dir = Path(output_dir)
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = REQUIRED_ARTIFACTS - actual
    extra = actual - REQUIRED_ARTIFACTS
    if missing:
        raise ValueError(f"missing artifacts: {sorted(missing)}")
    if extra:
        raise ValueError(f"unexpected artifacts: {sorted(extra)}")
    complete = json.loads((output_dir / "COMPLETE").read_text(encoding="utf-8"))
    expected_hashes = complete.get("sha256", {})
    if set(expected_hashes) != REQUIRED_ARTIFACTS - {"COMPLETE"}:
        raise ValueError("COMPLETE hash inventory is incomplete")
    for name, expected in expected_hashes.items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    result = {
        "manifest": json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        ),
        "tests": json.loads((output_dir / "tests.json").read_text(encoding="utf-8")),
        "aggregate": json.loads(
            (output_dir / "aggregate.json").read_text(encoding="utf-8")
        ),
    }
    for module in _JSONL_MODULES:
        result[module] = [
            json.loads(line)
            for line in (output_dir / f"{module}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
    _validate_result_contract(result)
    _scan_sensitive(result)
    for module in ("routing", "evidence", "memory"):
        if (output_dir / f"{module}.csv").read_text(
            encoding="utf-8"
        ) != render_csv(result[module]):
            raise ValueError(f"{module}.csv does not regenerate from canonical JSON")
    if (output_dir / "report.md").read_text(
        encoding="utf-8"
    ) != render_report(result["aggregate"]):
        raise ValueError("report.md does not regenerate from aggregate JSON")
    if (output_dir / "paper-table.md").read_text(
        encoding="utf-8"
    ) != regenerate_paper_table(output_dir / "aggregate.json"):
        raise ValueError("paper table does not regenerate from aggregate JSON")


def _scan_sensitive(value: object) -> None:
    for key in _iter_keys(value):
        if re.fullmatch(
            r"(?:api[_-]?key|secret(?:[_-]?key)?|password|credential|access[_-]?token|auth[_-]?token|private[_-]?key)",
            key,
            flags=re.IGNORECASE,
        ):
            raise ValueError("artifact contains a secret-bearing field")
    strings = list(_iter_strings(value))
    text = "\n".join(strings)
    lowered = text.lower()
    patterns = (
        r"(?:api[_-]?key|secret|password|token)\s*[=:]\s*[^\s\"']+",
        r"AKIA[0-9A-Z]{16}",
        r"[A-Za-z]:\\[^\r\n\"']+",
        r"/(?:home|Users|tmp|private|var|opt|mnt)/[^\r\n\"']+",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
        raise ValueError(
            "artifact contains sensitive credential or private path material"
        )
    for private in (os.getenv("USERNAME", ""), platform.node()):
        if len(private) >= 4 and private.lower() in lowered:
            raise ValueError("artifact contains private username or hostname")
    inherited_secret_name = re.compile(
        r"(?:credential|token|secret|password|(?:^|_)key(?:_|$))",
        flags=re.IGNORECASE,
    )
    inherited_secret_values = {
        secret
        for name, secret in os.environ.items()
        if secret and inherited_secret_name.search(name)
    }
    for secret in inherited_secret_values:
        if any(secret in item for item in strings):
            raise ValueError("artifact contains an inherited credential value")


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _iter_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_keys(item)


def normalized_reproducibility_bytes(result_path: Path) -> bytes:
    """Canonicalize non-timing results, removing only timestamp and timing values."""
    result_path = Path(result_path)
    if result_path.is_dir():
        payload = {
            "manifest": json.loads(
                (result_path / "manifest.json").read_text(encoding="utf-8")
            ),
            "routing": [
                json.loads(line)
                for line in (result_path / "routing.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ],
            "evidence": [
                json.loads(line)
                for line in (result_path / "evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ],
            "memory": [
                json.loads(line)
                for line in (result_path / "memory.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ],
            "tests": json.loads(
                (result_path / "tests.json").read_text(encoding="utf-8")
            ),
            "aggregate": json.loads(
                (result_path / "aggregate.json").read_text(encoding="utf-8")
            ),
        }
    else:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    if "manifest" in payload:
        payload["manifest"].pop("timestamp_utc", None)
    aggregate = payload.get("aggregate", payload)
    if isinstance(aggregate.get("run"), dict):
        aggregate["run"].pop("timestamp_utc", None)
    aggregate.pop("timing", None)
    return canonical_json_bytes(payload)


def write_provenance(
    aggregate_path: Path, artifact_commit: str, output_path: Path
) -> None:
    """Write non-self-referential source/artifact commit and hash lineage."""
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", artifact_commit):
        raise ValueError("artifact commit must be a full hexadecimal SHA")
    aggregate_path, output_path = Path(aggregate_path), Path(output_path)
    artifact_dir = aggregate_path if aggregate_path.is_dir() else aggregate_path.parent
    aggregate_path = artifact_dir / "aggregate.json"
    validate_artifacts(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    complete_path = artifact_dir / "COMPLETE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_hash = sha256_file(aggregate_path)
    if complete["sha256"].get("aggregate.json") != aggregate_hash:
        raise ValueError("aggregate hash does not match COMPLETE")
    aggregate_source = aggregate.get("run", {}).get("source_commit")
    if aggregate_source != manifest["source_commit"]:
        raise ValueError("aggregate and manifest source commits contradict")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_source_commit": manifest["source_commit"],
        "artifact_commit": artifact_commit.lower(),
        "aggregate_sha256": aggregate_hash,
        "artifact_hashes": complete["sha256"],
        "fixture_hashes": manifest["fixture_hashes"],
        "code_hashes": manifest["code_hashes"],
        "config_hashes": manifest["config_hashes"],
    }
    if output_path.exists():
        raise FileExistsError(f"provenance output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp")
    temp.write_bytes(canonical_json_bytes(payload))
    os.replace(temp, output_path)


def _git_source_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return commit, dirty


def build_manifest(
    fixture_dir: Path,
    test_manifest: Path,
    project_root: Path,
    timing_repeats: int,
    network_attempts: dict,
    *,
    source_state: tuple[str, bool] | None = None,
) -> dict:
    source_commit, source_dirty = source_state or _git_source_state(project_root)
    fixture_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path(fixture_dir).iterdir())
        if path.is_file()
    }
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("network.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("policies.py"),
        Path(__file__).with_name("schema.py"),
        Path(__file__).parent.parent / "run_offline_evidence.py",
        project_root / "apps/chat/consumers/chat_receive.py",
        project_root / "apps/chat/services/rag_intent.py",
        project_root / "apps/chat/services/rag_query.py",
        project_root / "apps/chat/services/rag_evidence.py",
        project_root / "apps/chat/services/rag_config.py",
        project_root / "apps/chat/services/rag_pipeline.py",
        project_root / "aquillm/memory.py",
        project_root / "lib/llm/types/conversation.py",
        project_root / "lib/llm/types/messages.py",
        project_root / "lib/llm/types/tools.py",
        project_root / "lib/memory/extraction/stable_facts.py",
    ]
    dependencies = {}
    for distribution in (
        "Django",
        "PyYAML",
        "pydantic",
        "pytest",
        "requests",
        "asgiref",
        "structlog",
    ):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "unavailable"
    config_hash = hashlib.sha256(canonical_json_bytes(CANONICAL_ENV)).hexdigest()
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"offline-{source_commit[:12]}",
        "timestamp_utc": now,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "fixture_hashes": fixture_hashes,
        "fixture_sensitivity": "synthetic_public",
        "code_hashes": {
            path.resolve().relative_to(project_root.resolve()).as_posix(): sha256_file(
                path
            )
            for path in code_paths
        },
        "config_hashes": {"canonical_env": config_hash},
        "canonical_env": dict(CANONICAL_ENV),
        "environment": {
            "os": platform.system(),
            "processor": platform.machine() or "unknown",
            "python": platform.python_version(),
            "timing_repeats": timing_repeats,
            "timing_warmups": 1,
            "evidence_timing_candidate_sizes": [1, 10, 100],
            "dependencies": dependencies,
        },
        "component_network_attempts": network_attempts,
        "test_manifest_hash": sha256_file(test_manifest),
    }


__all__ = [
    "CANONICAL_ENV",
    "REQUIRED_ARTIFACTS",
    "build_manifest",
    "normalized_reproducibility_bytes",
    "regenerate_paper_table",
    "run_component_evaluation",
    "run_test_manifest",
    "validate_artifacts",
    "write_artifacts",
    "write_provenance",
]
