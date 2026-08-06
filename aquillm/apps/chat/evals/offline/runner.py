"""Deterministic, network-denied offline component evaluation and artifacts."""

from __future__ import annotations

import asyncio
import copy
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
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
        expected_direct = case["gold"].get(
            "direct_pipeline_action", case["gold"]["production_action"]
        )
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
        checks = {
            "classifier": actual["classifier"] == expected["classifier"],
            "reason": actual["reason"] == expected["reason"],
            "helper_action": helper_action == expected["helper_action"],
            "direct_action": direct_action == expected_direct,
            "query": expected_query is None or query.strip() == expected_query.strip(),
        }
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "module": "routing",
                "case_id": case["id"],
                "stratum": case["stratum"],
                "expected": expected,
                "actual": actual,
                "conformant": all(checks.values()),
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

    def fail_immediately(*_args, **_kwargs):
        raise RuntimeError("controlled offline extraction failure")

    def persist(_user, facts):
        persisted.append(list(facts))

    def execute(case):
        before = len(persisted)
        start = time.perf_counter()
        count = promote_profile_facts_for_turn(
            7,
            case["input"]["user_content"],
            case["input"]["assistant_content"],
        )
        elapsed = time.perf_counter() - start
        return count, persisted[before], elapsed

    with (
        patch("aquillm.memory.User.objects.filter") as lookup,
        patch("aquillm.memory._promote_profile_facts", side_effect=persist),
        patch(
            "lib.memory.extraction.stable_facts.requests.post",
            side_effect=fail_immediately,
        ),
    ):
        lookup.return_value.first.return_value = fake_user
        explicit_count, explicit_facts, explicit_latency = execute(explicit)
        heuristic_count, heuristic_facts, heuristic_latency = execute(heuristic)
    return {
        "orchestration_failure": "controlled_immediate_extraction_failure",
        "explicit_remember": {
            "branch": "explicit_remember",
            "fact_count": explicit_count,
            "facts": explicit_facts,
            "latency_seconds": explicit_latency,
        },
        "heuristic": {
            "branch": "heuristic",
            "fact_count": heuristic_count,
            "facts": heuristic_facts,
            "latency_seconds": heuristic_latency,
        },
        "network_failure_latency_seconds": explicit_latency + heuristic_latency,
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
    evidence_case = datasets["evidence"]["cases"][0]
    memory_case = datasets["memory"]["cases"][0]
    return [
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
            "evidence",
            lambda: build_evidence_packet(
                {"result": evidence_case["candidates"]},
                query=evidence_case["question"],
                search_scope="synthetic fixtures",
                token_budget=evidence_case["token_budget"],
            ),
            repeats,
            {"candidate_count": len(evidence_case["candidates"])},
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


def _aggregate(routing: list[dict], evidence: list[dict], memory: list[dict], timings):
    classifier = {}
    for field in _CLASSIFIER_FIELDS:
        classifier[field] = binary_metrics(
            [record["expected"]["classifier"][field] for record in routing],
            [record["actual"]["classifier"][field] for record in routing],
        )
    reason = categorical_conformance(
        [record["expected"]["reason"] for record in routing],
        [record["actual"]["reason"] for record in routing],
        labels=list(_REASONS),
    )
    helper_action = categorical_conformance(
        [record["expected"]["helper_action"] for record in routing],
        [record["actual"]["helper_action"] for record in routing],
        labels=list(_ACTIONS),
    )
    direct_action = categorical_conformance(
        [record["expected"]["direct_action"] for record in routing],
        [record["actual"]["direct_action"] for record in routing],
        labels=list(_ACTIONS),
    )
    query_records = [
        record for record in routing if record["expected"]["query"] is not None
    ]
    query = query_conformance(
        [record["expected"]["query"] for record in query_records],
        [record["actual"]["query"] for record in query_records],
    )
    policy_pairs = []
    for record in evidence:
        pair = {
            "aquillm": dict(record["aquillm"]),
            "sequential": dict(record["sequential"]),
        }
        for policy in ("aquillm", "sequential"):
            pair[policy].update(pair[policy]["citation_diagnostics"])
        policy_pairs.append(pair)
    comparison_metrics = (
        "relevant_evidence_recall",
        "relevant_document_coverage",
        "distinct_selected_documents",
        "estimated_token_use",
        "overrun_tokens",
        "duplicate_count",
        "conflict_count",
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
            "aquillm": aggregate_evidence([record["aquillm"] for record in evidence]),
            "sequential": aggregate_evidence(
                [record["sequential"] for record in evidence]
            ),
            "paired_comparisons": {
                metric: compare_policies(policy_pairs, metric)
                for metric in comparison_metrics
            },
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
        "timing": {record["module"]: record for record in timings},
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


def run_test_manifest(path: Path, project_root: Path) -> dict:
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
    env = os.environ.copy()
    env.pop("DJANGO_SETTINGS_MODULE", None)
    env.pop("PYTEST_ADDOPTS", None)
    env.update(
        {
            "SECRET_KEY": "offline-test-only",
            "GOOGLE_OAUTH2_CLIENT_ID": "offline-test-only",
            "GOOGLE_OAUTH2_CLIENT_SECRET": "offline-test-only",
            "OPENAI_API_KEY": "offline-test-only",
            "GEMINI_API_KEY": "offline-test-only",
            "ANTHROPIC_API_KEY": "offline-test-only",
        }
    )
    with tempfile.TemporaryDirectory(prefix="aquillm-junit-") as temp:
        junit = Path(temp) / "junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *[entry["node_id"] for entry in included],
            "--junitxml",
            str(junit),
            "-q",
        ]
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        counts, outcomes = _parse_junit(junit, len(included))
    entries = []
    outcome_index = 0
    for entry in data["entries"]:
        item = dict(entry)
        if entry["status"] == "included":
            item["outcome"] = outcomes[outcome_index]
            outcome_index += 1
        else:
            item["outcome"] = "unavailable"
        entries.append(item)
    counts["unavailable"] = len(data["entries"]) - len(included)
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "summary": counts,
        "exit_code": completed.returncode,
        "stdout": _redact_subprocess_output(completed.stdout),
        "stderr": _redact_subprocess_output(completed.stderr),
        "command": [
            "python",
            "-m",
            "pytest",
            "<exact-manifest-nodes>",
            "--junitxml",
            "<temporary>",
        ],
        "network_scope": "component_execution_only",
        "declared_network_policy": "no_network",
    }


def _parse_junit(path: Path, expected: int) -> tuple[dict, list[str]]:
    if not path.is_file():
        return (
            {
                "collected": expected,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": expected,
            },
            ["error"] * expected,
        )
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    passed = collected - failed - errors - skipped
    outcomes = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None:
            outcomes.append("failed")
        elif case.find("error") is not None:
            outcomes.append("error")
        elif case.find("skipped") is not None:
            outcomes.append("skipped")
        else:
            outcomes.append("passed")
    if len(outcomes) < expected:
        outcomes.extend(["error"] * (expected - len(outcomes)))
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }, outcomes[:expected]


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


def _write_csv(path: Path, records: list[dict]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def regenerate_paper_table(aggregate_path: Path) -> str:
    """Render the paper table using aggregate JSON as the sole data source."""
    aggregate = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))

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
            ratio(nested("evidence", "aquillm", "macro_relevant_evidence_recall")),
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


def _report_text(aggregate: dict) -> str:
    return (
        "# Preliminary offline component evaluation report\n\n"
        "This report separates deterministic component conformance, contract-test "
        "counts, and local microbenchmarks. Fixed-set misses remain visible and are "
        "not integrity failures.\n\n"
        "## Excluded claims\n\n"
        + "".join(f"- {claim}\n" for claim in aggregate["excluded_claims"])
    )


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
            _write_csv(temp_dir / f"{module}.csv", payload[module])
        (temp_dir / "report.md").write_text(
            _report_text(payload["aggregate"]), encoding="utf-8", newline="\n"
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
    manifest_path = aggregate_path.parent / "manifest.json"
    complete_path = aggregate_path.parent / "COMPLETE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_source_commit": manifest["source_commit"],
        "artifact_commit": artifact_commit.lower(),
        "aggregate_sha256": sha256_file(aggregate_path),
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
) -> dict:
    source_commit, source_dirty = _git_source_state(project_root)
    fixture_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path(fixture_dir).iterdir())
        if path.is_file()
    }
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("network.py"),
        project_root / "apps/chat/services/rag_intent.py",
        project_root / "apps/chat/services/rag_query.py",
        project_root / "apps/chat/services/rag_evidence.py",
        project_root / "apps/chat/services/rag_pipeline.py",
        project_root / "lib/memory/extraction/stable_facts.py",
    ]
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
        "code_hashes": {path.name: sha256_file(path) for path in code_paths},
        "config_hashes": {"canonical_env": config_hash},
        "canonical_env": dict(CANONICAL_ENV),
        "environment": {
            "os": platform.system(),
            "processor": platform.machine() or "unknown",
            "python": platform.python_version(),
            "timing_repeats": timing_repeats,
            "timing_warmups": 1,
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
