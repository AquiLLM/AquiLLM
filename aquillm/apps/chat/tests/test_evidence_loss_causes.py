"""Loss explanations require ordered, affected-source exclusion observations."""

import pytest

from apps.chat.evals.evidence_bounded_partial import bounded_partial
from apps.chat.evals.evidence_quality_eval import text_digest
from apps.chat.tests.test_evidence_bounded_partial import partial_case


def loss_case():
    case, row, checks = partial_case()
    source = {
        "source_id": "lost",
        "revision": "r",
        "text": "Blair owns release.",
        "authorized_at_answer": True,
    }
    span = {
        "source_id": "lost",
        "revision": "r",
        "text": source["text"],
        "fingerprint": text_digest(source["text"]),
        "start": 0,
        "end": len(source["text"]),
    }
    case["sources"].append(source)
    row["source_bindings"].append(
        {"document_id": "lost-doc", "chunk_id": 2, "source_id": "lost", "revision": "r"}
    )
    row["events"][0]["rows"].append(
        {
            "doc_id": "lost-doc",
            "chunk_id": 2,
            "text": source["text"],
            "citation": "[doc:lost-doc chunk:2]",
        }
    )
    checks["available_facts"].append(
        {"fact_id": "lost", "span": span, "acquisition_event": 0}
    )
    checks["usable_fact_count"] = 2
    checks["losses"] = [
        {
            "fact_id": "lost",
            "sdk_event": 1,
            "reason": "authorization denied",
            "reviewed": True,
        }
    ]
    return case, row, checks


def test_missing_observed_cause_cannot_be_supplied_by_human_reason():
    case, row, checks = loss_case()
    assert not bounded_partial(case, row, checks)


def observed_selection(row):
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_types import (
        SelectionCandidate,
        SelectionLimits,
        SelectionProfile,
    )
    from lib.evidence_observation import observe
    from lib.retrieval.evidence import (
        PreparedEvidence,
        SourceEvidence,
        SourceSpan,
        fingerprint_prepared_evidence,
    )

    candidates = []
    for rank, packet in enumerate(row["events"][0]["rows"], 1):
        fingerprint = text_digest(packet["text"])
        source = SourceEvidence(
            packet["chunk_id"], packet["doc_id"], 0, fingerprint, packet["text"]
        )
        span = SourceSpan(
            source.chunk_id, fingerprint, 0, len(source.text), source.text
        )
        prepared = PreparedEvidence(
            source,
            (span,),
            fingerprint_prepared_evidence(source, (span,)),
            8,
            "complete",
        )
        candidates.append(
            SelectionCandidate(
                source.chunk_id,
                source.document_id,
                0,
                source.text,
                1 / rank,
                rank,
                fingerprint,
                packet,
                prepared,
                8,
            )
        )
    events = []
    with observe(lambda event, data: events.append({"event": event, **data})):
        selected = select_evidence(
            candidates,
            profile=SelectionProfile("p", 1, 0, "1"),
            limits=SelectionLimits(1, 1, 100),
            score_status="rank_fallback",
        )
    assert selected.candidates == (candidates[0],)
    exclusion = next(e for e in events if e["event"] == "selection_excluded")
    return exclusion


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "kind",
        "source",
        "phase",
        "order",
        "resource",
        "not_exhausted",
        "fingerprint",
        "span",
    ],
)
def test_actual_selection_loss_cause_is_required_and_checked(mutation):
    case, row, checks = loss_case()
    event = observed_selection(row)
    row["events"].insert(1, event)
    loss = checks["losses"][0]
    loss.update(
        sdk_event=2,
        reason="passage capacity exhausted",
        cause={"type": "selection_limit", "event": 1, "resource": "passages"},
    )
    assert bounded_partial(case, row, checks)
    if mutation is None:
        return
    if mutation == "kind":
        loss["cause"]["type"] = "authorization_denied"
    if mutation == "source":
        event["source"]["document_id"] = "other"
    if mutation == "phase":
        event["phase"] = "acquisition"
    if mutation == "order":
        loss["cause"]["event"] = 2
    if mutation == "resource":
        loss["cause"]["resource"] = "pairs"
    if mutation == "not_exhausted":
        event["used"] = 0
    if mutation == "fingerprint":
        event["source"]["fingerprint"] = "stale"
    if mutation == "span":
        event["source"]["spans"] = [[1, 2]]
    assert not bounded_partial(case, row, checks)
