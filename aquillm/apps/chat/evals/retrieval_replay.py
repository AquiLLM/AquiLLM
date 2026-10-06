"""Sequential, retrieval-only replay; raw observations never leave this process."""

import hashlib
import json
import os
import re
from contextlib import ExitStack, contextmanager, nullcontext
from threading import Event, Lock
from unittest.mock import patch

EXPERIMENTS = (
    "baseline",
    "graph-off",
    "direct-only",
    "extended-only",
    "depth30",
    "depth60",
    "full-text",
    "depth60-full",
)


class MappingError(ValueError):
    pass


class GenerationDenied(RuntimeError):
    pass


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("manifest_unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("manifest_version")
    questions = manifest.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 100:
        raise ValueError("manifest_questions")
    seen = set()
    for row in questions:
        if not isinstance(row, dict) or set(row) - {"id", "question", "targets"}:
            raise ValueError("manifest_question_fields")
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", identifier)
            or identifier in seen
        ):
            raise ValueError("manifest_question_id")
        seen.add(identifier)
        if (
            not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or len(row["question"]) > 20000
        ):
            raise ValueError("manifest_question_text")
        targets = row.get("targets", [])
        if not isinstance(targets, list) or len(targets) > 100:
            raise ValueError("manifest_targets")
        for target in targets:
            validate_target(target)
    return questions


def validate_target(target):
    if not isinstance(target, dict) or set(target) - {
        "source_sha256",
        "text_sha256",
        "answer_sha256",
        "start",
        "end",
    }:
        raise ValueError("target_fields")
    for key in ("source_sha256", "text_sha256", "answer_sha256"):
        if key in target and (
            not isinstance(target[key], str)
            or not re.fullmatch("[0-9a-f]{64}", target[key])
        ):
            raise ValueError("target_fingerprint")
    if "source_sha256" not in target or not (
        "text_sha256" in target or "answer_sha256" in target
    ):
        raise ValueError("target_identity")
    if "answer_sha256" in target:
        start, end = target.get("start"), target.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("target_offsets")
    elif "start" in target or "end" in target:
        raise ValueError("target_offsets")


def resolve_targets(targets, sources):
    resolved = []
    for target in targets:
        validate_target(target)
        matches = [
            source
            for source in sources
            if fingerprint(source["source"]) == target["source_sha256"]
        ]
        if len(matches) != 1:
            raise MappingError("source_missing_or_ambiguous")
        source = matches[0]
        chunks = source["chunks"]
        answer = None
        if "answer_sha256" in target:
            answer = source["source"][target["start"] : target["end"]]
            if not answer or fingerprint(answer) != target["answer_sha256"]:
                raise MappingError("answer_span_mismatch")
            chunks = [
                chunk
                for chunk in chunks
                if chunk["start"] <= target["start"]
                and chunk["end"] >= target["end"]
                and answer in chunk["text"]
            ]
        if "text_sha256" in target:
            chunks = [
                chunk
                for chunk in chunks
                if fingerprint(chunk["text"]) == target["text_sha256"]
            ]
            if len(chunks) > 1:
                raise MappingError("chunk_ambiguous")
        if not chunks:
            raise MappingError("supporting_chunk_missing")
        resolved.append(
            {
                **target,
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "answer": answer,
            }
        )
    return resolved


class GenerationGuard:
    def __init__(self, interface):
        self._interface = interface
        self.attempted = False

    def __getattr__(self, name):
        value = getattr(self._interface, name)
        if callable(value):

            def denied(*args, **kwargs):
                self.attempted = True
                raise GenerationDenied("generation_prohibited")

            return denied
        # Never expose nested clients that could bypass the interface guard.
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise GenerationDenied("interface_attribute_prohibited")


class EventRecorder:
    def __init__(self):
        self.events = []
        self._closed = False
        self._lock = Lock()

    def __call__(self, name, data):
        with self._lock:
            if not self._closed:
                self.events.append((name, data))

    def close(self):
        with self._lock:
            self._closed = True


@contextmanager
def replay_runtime(experiment, settings, config, pipeline, recorder):
    if experiment not in EXPERIMENTS:
        raise ValueError("experiment_unknown")
    env = {
        "RAG_DIRECT_ENABLED": "1",
        "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_EVIDENCE_TEXT_MODE": "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
    }
    attributes = [
        (settings, "RAG_CACHE_ENABLED", False),
        (pipeline, "synthesize_from_evidence", recorder),
    ]
    if experiment == "graph-off":
        attributes.append((settings, "KG_OVERLAY_ENABLED", False))
    if experiment in ("direct-only", "extended-only"):
        attributes.extend(
            [
                (settings, "KG_OVERLAY_ENABLED", True),
                (settings, "KG_GRAPH_DIRECT_ENABLED", experiment == "direct-only"),
                (settings, "KG_GRAPH_EXTENDED_ENABLED", experiment == "extended-only"),
            ]
        )
        env.update(
            KG_GRAPH_DIRECT_ENABLED=str(int(experiment == "direct-only")),
            KG_GRAPH_EXTENDED_ENABLED=str(int(experiment == "extended-only")),
        )
        if experiment == "extended-only":
            env["KG_DIRECT_EMBEDDING_ENABLED"] = "0"
            attributes.append((settings, "KG_DIRECT_EMBEDDING_ENABLED", False))
    if experiment in ("depth30", "depth60", "depth60-full"):
        depth = 30 if experiment == "depth30" else 60
        attributes.extend(
            [
                (config, "vector_top_k", depth),
                (config, "trigram_top_k", depth),
                (settings, "RAG_CANDIDATE_MULTIPLIER", 4.0),
            ]
        )
    if experiment in ("full-text", "depth60-full"):
        env["APP_RERANK_DOC_CHAR_LIMIT"] = "2048"
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env))
        for obj, name, value in attributes:
            stack.enter_context(patch.object(obj, name, value, create=True))
        yield


def _identity(row):
    return row.get("chunk_id", row.get("id", row.get("i")))


def _text(row):
    return row.get("text") or row.get("x") or row.get("content") or ""


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


def turn_succeeded(
    outcome, *, packet_recorded, observer_failed, sdk_started, generation_attempted
):
    return (
        outcome == "handled"
        and packet_recorded
        and not (observer_failed or sdk_started or generation_attempted)
    )


async def run_turn(
    question, targets, user, collections, experiment, settings, config, pipeline
):
    from time import perf_counter
    from types import SimpleNamespace

    from apps.chat.refs import CollectionsRef
    from lib.evidence_observation import observe
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import UserMessage

    recorder, guard = EventRecorder(), GenerationGuard(config.llm_interface)
    search_failed = Event()
    original_search = getattr(pipeline, "_run_vector_search", None)

    def record_search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        if not isinstance(result, dict) or "exception" in result:
            search_failed.set()
        return result

    packet, recorded = [], False

    async def record_packet(llm, conversation, evidence, stream_func=None):
        nonlocal packet, recorded
        packet, recorded = list(evidence.chunks), True
        return conversation

    def deny_rewrite(*args, **kwargs):
        guard.attempted = True
        raise GenerationDenied("generation_prohibited")

    conversation = Conversation(
        system="", messages=[UserMessage(content=question["question"])]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef(collections), convo=conversation
    )
    start, outcome, error = perf_counter(), None, None
    from apps.chat.services import rag_query

    with (
        replay_runtime(experiment, settings, config, pipeline, record_packet),
        patch.object(config, "llm_interface", guard),
        patch.object(rag_query, "_llm_rewrite_query", deny_rewrite),
        patch.object(pipeline, "_run_vector_search", record_search)
        if original_search is not None
        else nullcontext(),
    ):
        with observe(recorder) as observation:
            try:
                outcome = await pipeline.run_direct_rag_turn(
                    consumer, guard, conversation, stream_func=None
                )
            except Exception as exc:
                error = type(exc).__name__
            finally:
                recorder.close()
    result = summarize(recorder.events, packet, targets)
    statuses = result.get("search_statuses", [])
    complete = all(
        key in {name for name, _ in recorder.events}
        for key in (
            "rag_prepared_queries",
            "rag_search_outcomes",
            "rag_final_selection",
        )
    )
    success = (
        turn_succeeded(
            outcome,
            packet_recorded=recorded,
            observer_failed=observation.failed.is_set(),
            sdk_started=observation.sdk_started.is_set(),
            generation_attempted=guard.attempted,
        )
        and error is None
        and not search_failed.is_set()
        and complete
        and bool(statuses)
        and all(status == "ok" for status in statuses)
    )
    result.update(
        id=question["id"],
        success=success,
        outcome=outcome if outcome in ("handled", "skipped") else "unexpected",
        error_type=error,
        observer_failed=observation.failed.is_set(),
        sdk_started=observation.sdk_started.is_set(),
        generation_attempted=guard.attempted,
        search_result_failed=search_failed.is_set(),
        elapsed_ms=round((perf_counter() - start) * 1000, 3),
    )
    return result
