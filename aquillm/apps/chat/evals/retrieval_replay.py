"""Sequential, retrieval-only replay; raw observations never leave this process."""

import json
import re
from contextlib import nullcontext
from threading import Event, Lock
from unittest.mock import patch

from .replay_controls import EXPERIMENTS as EXPERIMENTS
from .replay_controls import replay_runtime
from .replay_observations import fingerprint, summarize


class MappingError(ValueError):
    pass


class GenerationDenied(RuntimeError):
    pass


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
        # Proposal observation needs metadata, never the provider's nested clients.
        self.base_args = {"model": "unknown"}

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


def turn_succeeded(
    outcome, *, packet_recorded, observer_failed, sdk_started, generation_attempted
):
    return (
        outcome == "handled"
        and packet_recorded
        and not (observer_failed or sdk_started or generation_attempted)
    )


async def run_turn(
    question,
    targets,
    user,
    collections,
    experiment,
    settings,
    config,
    pipeline,
    **modes,
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
        # Preservation acquisition has action events, rather than legacy search
        # events. Emit equivalent replay-only observations at the real boundary.
        from apps.chat.services.rag_config import rag_preservation_config
        from apps.chat.services.retrieval_query_runtime import query_proposals_enabled

        if rag_preservation_config().active or query_proposals_enabled():
            from lib.replay_observation import prepared_queries, search_outcomes

            prepared_queries([args[1]], args[2], args[2])
            search_outcomes([args[1]], [result])
        return result

    packet, recorded, retrieval_status = [], False, None

    async def record_packet(llm, conversation, evidence, stream_func=None):
        nonlocal packet, recorded, retrieval_status
        packet, recorded = list(evidence.chunks), True
        retrieval_status = getattr(evidence, "retrieval_status", "results_found")
        from apps.chat.services.rag_config import (
            direct_rag_top_k,
            evidence_token_budget,
        )
        from lib.replay_observation import final_selection

        final_selection(evidence, direct_rag_top_k(), evidence_token_budget)
        return conversation

    async def deny_rewrite(*args, **kwargs):
        guard.attempted = True
        raise GenerationDenied("generation_prohibited")

    conversation = Conversation(
        system="", messages=[UserMessage(content=question["question"])]
    )
    consumer = SimpleNamespace(
        user=user, col_ref=CollectionsRef(collections), convo=conversation
    )
    start, outcome, error = perf_counter(), None, None
    from apps.chat.services import retrieval_query_runtime

    with (
        replay_runtime(experiment, settings, config, pipeline, record_packet, **modes),
        patch.object(config, "llm_interface", guard),
        patch.object(retrieval_query_runtime, "_rewrite", deny_rewrite),
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
        and retrieval_status not in ("partial", "context_limited")
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
        retrieval_status=retrieval_status,
        elapsed_ms=round((perf_counter() - start) * 1000, 3),
    )
    return result
