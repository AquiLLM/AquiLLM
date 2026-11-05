"""Consume existing assessment pointers once, after ordinary source validation.

These pointers express the planner's unverified claims, never source authority or
semantic support. Only the normal prepared, completely scored pool can supply a
candidate. No retrieval, model call, or additional database read happens here.
"""

from dataclasses import dataclass
from hashlib import sha256
from os import getenv

from django.core.exceptions import ImproperlyConfigured

from lib.llm.turn_context import current_turn
from lib.retrieval.evidence import SourceEvidence

from .rag_coverage import CoverageAssessment, SupportReference
from .rag_selection import support_scores_usable

POLICY = "assessment-support-hints-v1"


def support_hints_enabled():
    name = "RAG_SELECTION_SUPPORT_HINTS_ENABLED"
    value = getenv(name, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ImproperlyConfigured(f"invalid {name}")


def _ranking_controls():
    from apps.documents.services.chunk_search_rank_policy import controls

    return tuple(sorted(controls().items()))


@dataclass(frozen=True, slots=True)
class _Hints:
    query_sha256: str
    runtime: object
    budget: object
    authorization: object
    turn: object
    frozen: object
    representation: object
    dense_manifest: str | None
    ranking_controls: tuple
    references: tuple[tuple[SupportReference, SourceEvidence], ...]


def _bind_selection_hints(runtime, question, assessment, sources):
    """Snapshot exact original views; retain no caller-provided source priority."""
    from apps.documents.services.source_loading import current_source_runtime

    with runtime.lock:
        previous = getattr(runtime, "_selection_support_hints", None)
        runtime._selection_support_hints = None
        if not support_hints_enabled() or previous is not None:
            return None
        frozen = getattr(runtime, "_frozen_retrieval_scope", None)
        turn = current_turn()
        if (current_source_runtime() is not runtime or not runtime.budget.can_publish()
                or (turn is not None and turn.budget is not runtime.budget)
                or frozen is None or type(assessment) is not CoverageAssessment
                or not 1 <= len(assessment.support) <= 4
                or not 1 <= len(assessment.requested_aspects) <= 4):
            return None
        views = tuple(sources)
        references = []
        for ref in assessment.support:
            if (type(ref) is not SupportReference or ref.aspect not in assessment.requested_aspects
                    or any(type(v) is not int for v in (ref.chunk_id, ref.start, ref.end))):
                return None
            matching = tuple(source for source in views if type(source) is SourceEvidence
                and source.chunk_id == ref.chunk_id and source.source_fingerprint == ref.source_fingerprint
                and 0 <= ref.start < ref.end <= len(source.text))
            if len(matching) != 1:
                return None
            references.append((ref, matching[0]))
        record = _Hints(sha256(question.encode()).hexdigest(), runtime, runtime.budget,
            runtime.authorization, turn, frozen, frozen.representation,
            getattr(frozen.dense, "manifest_sha256", None), _ranking_controls(), tuple(references))
        runtime._selection_support_hints = record
        return record


def bind_selection_hints(runtime, question, assessment, sources):
    from lib.evidence_observation import publish

    record = _bind_selection_hints(runtime, question, assessment, sources)
    publish("selection_support_hints", {"stage": "bound", "policy": POLICY,
        "references": len(record.references) if record else 0})
    return record


def clear_selection_hints(runtime, record):
    """An abandoned/custom handoff cannot leave hints for a later selection."""
    with runtime.lock:
        if getattr(runtime, "_selection_support_hints", None) is record:
            runtime._selection_support_hints = None


def _consume_selection_hints(runtime, prepared, pool, question):
    """Join all exact pointers or use ordinary selection; never refill evidence."""
    from apps.documents.services.source_loading import current_source_runtime

    if runtime is None:
        return ()
    with runtime.lock:
        record = getattr(runtime, "_selection_support_hints", None)
        runtime._selection_support_hints = None
        frozen = getattr(runtime, "_frozen_retrieval_scope", None)
        if (not support_hints_enabled() or type(record) is not _Hints
                or current_source_runtime() is not runtime or record.runtime is not runtime
                or record.budget is not runtime.budget or record.authorization is not runtime.authorization
                or record.turn is not current_turn() or record.frozen is not frozen
                or (record.turn is not None and record.turn.budget is not record.budget)
                or frozen is None or record.representation is not frozen.representation
                or record.dense_manifest != getattr(frozen.dense, "manifest_sha256", None)
                or record.ranking_controls != _ranking_controls()
                or record.query_sha256 != sha256(question.encode()).hexdigest()
                or not runtime.budget.can_publish() or prepared.fallback_reason is not None
                or not support_scores_usable(prepared.candidates, prepared.score_status)):
            return ()
        rows = {c.chunk_id: c for c in prepared.candidates}
        acquired_ids = [row.get("chunk_id", row.get("i")) for row in pool.rows]
        if (len(rows) != len(prepared.candidates) or len(set(acquired_ids)) != len(acquired_ids)
                or set(rows) != set(acquired_ids)):
            return ()
        grouped = {}
        for ref, original in record.references:
            candidate = rows.get(ref.chunk_id)
            evidence = candidate.prepared_evidence if candidate is not None else None
            if (evidence is None or evidence.source != original
                    or candidate.doc_id != original.document_id
                    or candidate.chunk_number != original.chunk_number
                    or candidate.source_fingerprint != original.source_fingerprint
                    or not any(span.chunk_id == ref.chunk_id
                        and span.source_fingerprint == ref.source_fingerprint
                        and span.start <= ref.start < ref.end <= span.end for span in evidence.spans)):
                return ()
            grouped.setdefault(ref.chunk_id, []).append(ref)
        return tuple((rows[pk], tuple(dict.fromkeys(refs))) for pk, refs in grouped.items())


def consume_selection_hints(runtime, prepared, pool, question):
    from lib.evidence_observation import publish

    hints = _consume_selection_hints(runtime, prepared, pool, question)
    publish("selection_support_hints", {"stage": "accepted", "policy": POLICY,
        "references": sum(len(refs) for _, refs in hints), "candidates": len(hints)})
    return hints
