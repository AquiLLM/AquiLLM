"""Observed support records plus named-human semantics for bounded answers."""

import json

from .evidence_loss_cause import observed_loss_cause
from .evidence_observation_json import normalize
from .evidence_quality_delivery import map_payload, parse_citations
from .evidence_quality_eval import identity, valid_spans


def bounded_partial(case, row, checks):
    try:
        return _bounded_partial(case, normalize(row), normalize(checks))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def _bounded_partial(case, row, checks):
    if not row.get("answer", "").strip() or not all(
        checks.get(k) is True
        for k in (
            "preserves_usable_support",
            "explicit_limit",
            "explicit_missing_aspects",
            "no_fabrication",
            "qualifications_correct",
            "citations_entailed",
            "losses_explained",
            "finalization_reserve_preserved",
        )
    ):
        return False
    limit = checks["limit_answer_span"]
    if (
        not 0 <= limit["start"] < limit["end"] <= len(row["answer"])
        or not limit["text"].strip()
        or row["answer"][limit["start"] : limit["end"]] != limit["text"]
    ):
        return False
    sources = {identity(s): s for s in case["sources"]}
    bindings = {
        (b["document_id"], b["chunk_id"]): sources[identity(b)]
        for b in row["source_bindings"]
    }
    sdk_events = [e for e in row["events"] if e["event"] == "sdk_start"]
    if not sdk_events or [e["payload"] for e in sdk_events] != row["sdk_payloads"]:
        return False
    valid = valid_spans(case, row["delivered"])
    if valid != row["delivered"]:
        return False
    hints = {
        key: [(s["start"], s["end"]) for s in valid if identity(s) == identity(source)]
        for key, source in bindings.items()
    }
    delivered, unknown = map_payload(row["sdk_payloads"][-1], bindings, hints)
    if unknown or delivered != row["delivered"]:
        return False
    facts = checks["available_facts"]
    ids = {f["fact_id"] for f in facts}
    retained = set(checks["retained_fact_ids"])
    if (
        len(ids) != len(facts)
        or len(retained) != len(checks["retained_fact_ids"])
        or not retained <= ids
        or type(checks.get("usable_fact_count")) is not int
        or type(checks.get("retained_fact_count")) is not int
        or checks["usable_fact_count"] != len(facts)
        or checks["retained_fact_count"] != len(retained)
    ):
        return False

    def contains(spans, span):
        return any(
            identity(s) == identity(span)
            and s["start"] <= span["start"]
            and s["end"] >= span["end"]
            for s in spans
        )

    for fact in facts:
        span = fact["span"]
        if not valid_spans(case, [span]):
            return False
        event = row["events"][fact["acquisition_event"]]
        if event["event"] != "retrieval_sources":
            return False
        acquired, unknown = map_payload(
            {"content": json.dumps({"result": event["rows"]})}, bindings
        )
        if unknown or not contains(acquired, span):
            return False
        if fact["fact_id"] in retained:
            start, end = fact["answer_start"], fact["answer_end"]
            portion = row["answer"][start:end]
            if (
                not contains(delivered, span)
                or not 0 <= start < end <= len(row["answer"])
                or not portion.strip()
                or fact["citation"] not in portion
                or identity(span)
                not in {identity(c) for c in parse_citations(portion, bindings)}
                or identity(span) not in {identity(c) for c in row["citations"]}
                or fact.get("entailed") is not True
                or fact.get("qualifications_correct") is not True
            ):
                return False
    losses = checks.get("losses", [])
    if {x["fact_id"] for x in losses} != ids - retained or len(losses) != len(
        ids - retained
    ):
        return False
    for loss in losses:
        # The acquired fact's absence is established by the final SDK packet;
        # semantics of the exclusion remain the bound human judgment.
        fact = next(f for f in facts if f["fact_id"] == loss["fact_id"])
        event = row["events"][loss["sdk_event"]]
        if (
            event != sdk_events[-1]
            or contains(delivered, fact["span"])
            or not loss.get("reason")
            or loss.get("reviewed") is not True
            or not observed_loss_cause(row, fact, loss, bindings)
        ):
            return False
    if facts:
        return bool(retained)
    refs = checks.get("empty_event_refs", [])
    return bool(
        not row.get("upstream")
        and not delivered
        and not row["citations"]
        and not parse_citations(row["answer"], bindings)
        and checks.get("empty_reason")
        and refs
        and all(
            row["events"][i].get("event") == "retrieval_sources"
            and row["events"][i].get("rows") == []
            for i in refs
        )
    )
