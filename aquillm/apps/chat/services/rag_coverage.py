"""Conservative coverage decisions and exact, revision-bound support references."""

import asyncio
import json
import unicodedata
from dataclasses import dataclass, replace

from .rag_planner_context import attempt_context, planner_views
from .rag_planner_contract import (
    LEAD_PLANNER_SYSTEM,
    PLANNER_SYSTEM,
    PlannerFailure,
    planner_contract,
    validate_action,
)
from .rag_planner_leads import EvidenceLead, evidence_leads_enabled, resolve_leads


def decide_answer_support(question, packet, *, assessment, calibration_id=None):
    from .rag_support_decision import decide_answer_support as decide

    return decide(
        question, packet, assessment=assessment, calibration_id=calibration_id
    )


@dataclass(frozen=True)
class AcquisitionAction:
    kind: str
    query: str = ""
    document_id: str | None = None
    chunk_id: int | None = None
    aspect: str = ""

    @property
    def signature(self):
        return json.dumps(
            [
                self.kind,
                " ".join(unicodedata.normalize("NFC", self.query).casefold().split()),
                self.document_id,
                self.chunk_id,
            ],
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class SupportReference:
    aspect: str
    chunk_id: int
    source_fingerprint: str
    start: int
    end: int


@dataclass(frozen=True)
class CoverageAssessment:
    requested_aspects: tuple[str, ...] = ()
    support: tuple[SupportReference, ...] = ()
    unresolved_aspects: tuple[str, ...] = ()
    next_action: AcquisitionAction | None = None
    certainty: str = "unknown"
    leads: tuple[EvidenceLead, ...] = ()


def needs_coverage_assessment(question, evidence_views, anchors):
    from .rag_lookup_coverage import lookup_coverage

    proof = lookup_coverage(question, evidence_views, anchors)
    return proof is None or bool(proof.unresolved_aspects)


def _aspects(value):
    if (
        not isinstance(value, list)
        or len(value) > 16
        or any(
            not isinstance(item, str) or not item.strip() or len(item) > 200
            for item in value
        )
    ):
        raise ValueError("invalid requested aspects")
    return tuple(dict.fromkeys(value))


def validate_assessment(payload, views, allowed_documents):
    if not isinstance(payload, dict) or set(payload) - {
        "requested_aspects",
        "support",
        "unresolved_aspects",
        "next_action",
    }:
        raise ValueError("invalid coverage schema")
    requested = _aspects(payload["requested_aspects"])
    unresolved = _aspects(payload["unresolved_aspects"])
    if not set(unresolved) <= set(requested):
        raise ValueError("unknown unresolved aspect")
    support = []
    records = payload.get("support", [])
    if not isinstance(records, list) or len(records) > 32:
        raise ValueError("invalid support list")
    for record in records:
        ref = SupportReference(**record)
        if ref.aspect not in requested or any(
            type(v) is not int for v in (ref.chunk_id, ref.start, ref.end)
        ):
            raise ValueError("invalid support reference")
        if not any(
            s.chunk_id == ref.chunk_id
            and s.source_fingerprint == ref.source_fingerprint
            and 0 <= ref.start < ref.end <= len(s.text)
            for s in views
        ):
            raise ValueError("support outside supplied source spans")
        support.append(ref)
    unresolved = tuple(
        dict.fromkeys(
            (
                *unresolved,
                *(a for a in requested if a not in {s.aspect for s in support}),
            )
        )
    )
    raw_action = payload.get("next_action")
    try:
        action = (
            validate_action(
                AcquisitionAction(**raw_action), unresolved, views, allowed_documents
            )
            if raw_action is not None
            else None
        )
    except (TypeError, ValueError):
        raise PlannerFailure("invalid_action") from None
    return CoverageAssessment(requested, tuple(support), unresolved, action, "assessed")


def recheck_support(assessment, delivered):
    spans = tuple(span for prepared in delivered for span in prepared.spans)
    kept = tuple(
        ref
        for ref in assessment.support
        if any(
            span.chunk_id == ref.chunk_id
            and span.source_fingerprint == ref.source_fingerprint
            and span.start <= ref.start < ref.end <= span.end
            for span in spans
        )
    )
    missing = tuple(
        dict.fromkeys(
            (
                *assessment.unresolved_aspects,
                *(ref.aspect for ref in assessment.support if ref not in kept),
            )
        )
    )
    return replace(
        assessment, support=kept, unresolved_aspects=missing, next_action=None, leads=()
    )


def _assess_with_leads(payload, views, allowed_documents, budget):
    """Validate the search action and the quoted ranking clues independently."""
    if not isinstance(payload, dict) or set(payload) != {
        "evidence_leads", "requested_aspects", "unresolved_aspects", "next_action"
    }:
        raise ValueError("invalid planner lead response")
    leads = resolve_leads(payload["evidence_leads"], views, budget)
    plan = {key: value for key, value in payload.items() if key != "evidence_leads"}
    requested = _aspects(plan["requested_aspects"])
    unresolved = _aspects(plan["unresolved_aspects"])
    if not 1 <= len(requested) <= 4 or len(unresolved) > 4:
        raise ValueError("invalid planner aspects")
    if leads:
        # A planner can give an outstanding detail a more specific name. With
        # no semantic support claimed, adding it only expands what is unknown.
        # Keep all named outstanding details inside the four-facet limit, then
        # carry broader labels where they fit. The original question is intact.
        plan["requested_aspects"] = list(dict.fromkeys((*unresolved, *requested)))[:4]
    try:
        assessment = validate_assessment(plan, views, allowed_documents)
    except PlannerFailure as exc:
        if not leads or exc.reason != "invalid_action":
            raise
        from lib.evidence_observation import publish

        publish("planner_action_rejected", {"reason": exc.reason})
        assessment = validate_assessment(
            dict(plan, next_action=None), views, allowed_documents
        )
    return replace(assessment, leads=leads)


async def assess_coverage(
    question,
    evidence_views,
    anchors,
    *,
    llm,
    budget,
    allowed_documents=(),
    attempts=(),
    latest_source_ids=(),
):
    """One bounded structured call to the same provider; never writes an answer."""
    evidence_views = tuple(evidence_views)
    views = planner_views(evidence_views, latest_source_ids=latest_source_ids)
    use_leads = evidence_leads_enabled()
    coordinates, response_format = planner_contract(
        views, allowed_documents, evidence_leads=use_leads
    )
    prompt = json.dumps(
        {
            "question": question,
            "unresolved_references": getattr(anchors, "unresolved_references", ()),
            "attempts": attempt_context(attempts, budget.limits.actions),
            "omitted_sources": len({s.identity for s in evidence_views}) - len(views),
            "omitted_documents": len(
                {s.document_id for s in evidence_views} - {s.document_id for s in views}
            ),
            "action_coordinates": coordinates,
            "evidence": [
                vars(v)
                if hasattr(v, "__dict__")
                else {
                    "chunk_id": v.chunk_id,
                    "document_id": v.document_id,
                    "source_fingerprint": v.source_fingerprint,
                    "text": v.text,
                }
                for v in views
            ],
        },
        ensure_ascii=False,
    )
    if not budget.reserve_text(len(prompt), kind="tokenized"):
        raise PlannerFailure("context_budget")
    from apps.chat.services.rag_context_budget import provider_context_capacity
    from lib.llm.evidence_guard import (
        ContextLimited,
        EvidenceProtection,
        protect_evidence,
    )

    context, margin = provider_context_capacity(llm)
    arguments = llm.base_args | {
        "system": LEAD_PLANNER_SYSTEM if use_leads else PLANNER_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": budget.limits.planner_output_tokens,
        "thinking_budget": 0,
    }
    if getattr(llm, "supports_json_schema", False):
        arguments["response_format"] = response_format
    try:
        with protect_evidence(
            EvidenceProtection(
                prompt,
                context,
                budget.limits.planner_output_tokens,
                margin,
                turn_budget=budget,
            )
        ):
            response = await asyncio.wait_for(
                llm.get_message(**arguments),
                timeout=min(budget.limits.planner_call_ms, budget.remaining_ms())
                / 1000,
            )
    except TimeoutError:
        raise PlannerFailure("timeout") from None
    except ContextLimited:
        raise PlannerFailure("context_budget") from None
    except Exception:
        raise PlannerFailure("provider_error") from None
    if getattr(response, "stop_reason", None) in {"length", "max_tokens"}:
        raise PlannerFailure("output_limit")
    if not isinstance(response.text, str) or not response.text.strip():
        raise PlannerFailure("empty_response")
    try:
        payload = json.loads(response.text)
    except (ValueError, TypeError):
        raise PlannerFailure("invalid_json") from None
    try:
        if use_leads:
            return _assess_with_leads(payload, views, allowed_documents, budget)
        return validate_assessment(payload, views, allowed_documents)
    except PlannerFailure:
        raise
    except (ValueError, TypeError, KeyError):
        raise PlannerFailure("invalid_schema") from None
