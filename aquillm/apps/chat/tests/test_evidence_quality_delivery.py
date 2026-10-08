"""Final SDK evidence, including legacy clipping, defines source delivery."""

from apps.chat.evals.evidence_quality_delivery import map_payload, parse_citations
from apps.chat.evals.evidence_quality_eval import text_digest


def source():
    return {
        "source_id": "s",
        "revision": "r",
        "text": "α full source tail μ",
        "authorized_at_answer": True,
    }


def test_legacy_sdk_prefix_is_only_prefix_and_ambiguous_mapping_unknown():
    import json

    s = source()
    rows = [
        {"doc_id": "d", "chunk_id": 1, "citation": "[doc:d chunk:1]", "text": "α full"}
    ]
    payload = {
        "messages": [
            {"role": "user", "content": "Tool result:\n" + json.dumps({"result": rows})}
        ]
    }
    delivered, unknown = map_payload(payload, {("d", 1): s})
    assert not unknown
    assert delivered == [
        {
            "source_id": "s",
            "revision": "r",
            "fingerprint": text_digest(s["text"]),
            "start": 0,
            "end": 6,
            "text": "α full",
        }
    ]
    s["text"] += " α full"
    assert map_payload(payload, {("d", 1): s})[1]


def test_source_multi_spans_require_exact_actual_sdk_text():
    s = source()
    payload = {
        "messages": [
            {
                "content": (
                    '{"result":[{"doc_id":"d","chunk_id":1,'
                    '"text":"α full ... μ","citation":"[doc:d chunk:1]"}]}'
                )
            }
        ]
    }
    hints = {("d", 1): [(0, 6), (len(s["text"]) - 1, len(s["text"]))]}
    delivered, unknown = map_payload(payload, {("d", 1): s}, hints)
    assert not unknown and len(delivered) == 2
    payload["messages"][0]["content"] = payload["messages"][0]["content"].replace(
        "μ", "x"
    )
    assert map_payload(payload, {("d", 1): s}, hints)[1]


def test_unknown_citation_is_retained_as_violation_identity():
    assert parse_citations("claim [doc:missing chunk:99]", {}) == [
        {"source_id": "unmapped:missing:99", "revision": "unknown"}
    ]


def test_known_legacy_marker_does_not_hide_exact_prefix():
    import json

    s = source()
    payload = {
        "content": json.dumps(
            {
                "result": [
                    {
                        "doc_id": "d",
                        "chunk_id": 1,
                        "citation": "[doc:d chunk:1]",
                        "text": "α full\n...[truncated for context window]...",
                    }
                ]
            }
        )
    }
    spans, unknown = map_payload(payload, {("d", 1): s})
    assert not unknown and spans[0]["end"] == 6


def test_sdk_split_support_after_excerpt_is_recorded_as_lost_despite_source_id():
    import json

    from apps.chat.evals.evidence_quality_eval import (
        evaluate,
        match_delivered,
        required_support,
    )

    text = "x" * 1200 + " threshold=7.3 ug/L"
    start = text.index("threshold=")
    s = {
        "source_id": "s",
        "revision": "r",
        "text": text,
        "authorized_at_answer": True,
    }
    case = {
        "case_id": "synthetic-split-support",
        "split": "development",
        "depth": "routine",
        "scenario": "retrieval",
        "quality_aggregation": "normal",
        "sources": [s],
        "gold_support": [
            {
                "support_id": "threshold",
                "source_id": "s",
                "revision": "r",
                "start": start,
                "end": len(text),
                "quote": text[start:],
            }
        ],
        "required_claims": [
            {"claim_id": "answer", "support_ids": ["threshold"], "labels": []}
        ],
        "expected_delivered_support_ids": ["threshold"],
        "permitted_citations": [{"source_id": "s", "revision": "r"}],
        "publishable_citations": [{"source_id": "s", "revision": "r"}],
    }
    excerpt = text[:1000] + "\n...[truncated for context window]..."
    payload = {
        "content": json.dumps(
            {
                "result": [
                    {
                        "doc_id": "d",
                        "chunk_id": 1,
                        "citation": "[doc:d chunk:1]",
                        "text": excerpt,
                    }
                ]
            }
        )
    }
    delivered, unknown = map_payload(payload, {("d", 1): s})
    assert not unknown
    assert delivered[0]["source_id"] == "s"
    assert match_delivered(case, delivered) == set()
    result = evaluate(
        case,
        {
            "delivered": delivered,
            "upstream": [
                {
                    "source_id": "s",
                    "revision": "r",
                    "fingerprint": text_digest(text),
                    "start": 0,
                    "end": len(text),
                    "text": text,
                }
            ],
            "citations": [{"source_id": "s", "revision": "r"}],
        },
    )
    assert required_support(case) == {"threshold"}
    assert result["support_recall"] == 0.0
    assert result["lost_before_sdk"] == ["threshold"]


def test_incomplete_sdk_or_pair_transport_accounting_cannot_claim_complete():
    from apps.chat.evals.evidence_quality_trace import Trace

    trace = Trace({})
    trace.events = [{"event": "sdk_start", "kind": "initial", "provider": "test"}]
    assert not trace.result("")["dispatch_accounting_complete"]
    trace.events.append({"event": "sdk_end"})
    assert trace.result("")["dispatch_accounting_complete"]
    trace.events.append({"event": "ledger", "operation": "start_pair", "result": True})
    assert not trace.result("")["dispatch_accounting_complete"]
    trace.events.append({"event": "rerank_http", "pairs": 1})
    assert trace.result("")["dispatch_accounting_complete"]
