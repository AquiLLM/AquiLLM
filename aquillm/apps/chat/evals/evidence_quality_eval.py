"""Exact final-delivery metrics. Source labels never manufacture human judgments."""

import hashlib
import json
import math
from pathlib import Path

MODES = ("baseline", "selection", "preservation", "combined")
FROZEN_SHA256 = "6eb8e3617ef8e07a083b33bc502d9615017a1a210997a2c333d3cd822ddc3b07"
HELDOUT_SHA256 = "01b3c9fed145c3ef46bf96552e1b4c0cc662260891af6033c114a2a2bac67395"


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value):
    return text_digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )


def identity(value):
    return value["source_id"], value["revision"]


def required_support(case):
    return {s for claim in case["required_claims"] for s in claim["support_ids"]}


def load_cases(path):
    raw = Path(path).read_bytes()
    root = json.loads(raw)
    cases = root["cases"]
    if root["offset_unit"] != "Python Unicode codepoints":
        raise ValueError("unsupported offset unit")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case")
    for c in cases:
        if c["split"] not in ("development", "heldout") or c["depth"] not in (
            "routine",
            "deeper",
        ):
            raise ValueError("invalid cohort")
        if c["quality_aggregation"] not in ("normal", "safety_only"):
            raise ValueError("invalid quality cohort")
        sources = {identity(s): s for s in c["sources"]}
        if len(sources) != len(c["sources"]):
            raise ValueError("duplicate source revision")
        gold = {g["support_id"]: g for g in c["gold_support"]}
        if len(gold) != len(c["gold_support"]):
            raise ValueError("duplicate support")
        for g in gold.values():
            source = sources[identity(g)]
            if (
                not (0 <= g["start"] < g["end"] <= len(source["text"]))
                or source["text"][g["start"] : g["end"]] != g["quote"]
            ):
                raise ValueError("gold is not an exact source slice")
        required = required_support(c)
        optional = set(c.get("optional_context_support_ids", []))
        if (
            not (required | optional | set(c["expected_delivered_support_ids"]))
            <= gold.keys()
            or required & optional
        ):
            raise ValueError("invalid mandatory/optional support")
        permitted = {identity(x) for x in c["permitted_citations"]}
        publishable = {identity(x) for x in c["publishable_citations"]}
        authorized = {key for key, s in sources.items() if s["authorized_at_answer"]}
        if not publishable <= permitted <= authorized or permitted & {
            identity(x) for x in c["forbidden_citations"]
        }:
            raise ValueError("inconsistent authorization labels")
    # The distributed corpus is byte-for-byte frozen, not a schema conversion.
    if Path(path).name == "evidence_quality_cases.json":
        if (
            hashlib.sha256(raw).hexdigest() != FROZEN_SHA256
            or digest([c for c in cases if c["split"] == "heldout"]) != HELDOUT_SHA256
        ):
            raise ValueError("frozen corpus changed")
    return cases


def valid_spans(case, delivered):
    sources = {identity(s): s for s in case["sources"]}
    result = []
    for span in delivered:
        source = sources.get(identity(span))
        if (
            source
            and source["authorized_at_answer"]
            and span.get("fingerprint") == text_digest(source["text"])
        ):
            start, end = span["start"], span["end"]
            if (
                0 <= start < end <= len(source["text"])
                and source["text"][start:end] == span["text"]
            ):
                result.append(span)
    return result


def match_delivered(case, delivered):
    spans = valid_spans(case, delivered)
    return {
        g["support_id"]
        for g in case["gold_support"]
        if any(
            identity(s) == identity(g)
            and s["start"] <= g["start"]
            and s["end"] >= g["end"]
            for s in spans
        )
    }


def support_recall(required, delivered):
    return len(set(required) & set(delivered)) / len(required) if required else None


def citation_violations(case, citations, delivered, *, frozen=True):
    allowed = (
        {identity(s) for s in valid_spans(case, delivered)}
        & {
            identity(s)
            for s in case["publishable_citations" if frozen else "permitted_citations"]
        }
        & {identity(s) for s in case["permitted_citations"]}
    )
    return [s for s in citations if identity(s) not in allowed]


def review_claims(case, review):
    claims = (
        review.get("claims", {})
        if review and review.get("kind") == "human" and review.get("reviewer")
        else {}
    )
    faithful, qualifications = [], []
    for claim in case["required_claims"]:
        labels = claims.get(claim["claim_id"], {})
        faithful.append(labels.get("faithful"))
        qualifications.extend(labels.get(k) for k in claim["labels"])

    def aggregate(values):
        return (
            sum(values) / len(values)
            if values and all(type(v) is bool for v in values)
            else None
        )

    answer_checks = (
        [review.get(k) for k in ("answer_faithful", "citation_entailment")]
        if claims
        else [None]
    )
    return {
        "faithfulness": aggregate(faithful + answer_checks),
        "qualification_accuracy": aggregate(qualifications),
        "review_unknowns": sum(
            type(x) is not bool for x in faithful + qualifications + answer_checks
        ),
    }


def ndcg(ranked, grades):
    if not grades:
        return None

    def dcg(values):
        return sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(values))

    ideal = dcg(sorted(grades.values(), reverse=True)[: len(ranked)])
    return dcg([grades.get(key, 0) for key in ranked]) / ideal if ideal else None


def score_safety(case, observation):
    if case["quality_aggregation"] != "safety_only":
        return None
    required = {"closed", "late_publications", "answer", "delivered", "citations"}
    if not required <= observation.keys():
        return None
    if observation["late_publications"] or not observation["closed"]:
        return False
    if case["scenario"] == "cancellation":
        return not any(observation[k] for k in ("answer", "delivered", "citations"))
    if any(
        k not in observation
        for k in ("limit_notice_reviewed", "stop_reason", "actions")
    ):
        return None
    return (
        observation["limit_notice_reviewed"] is True
        and observation["stop_reason"] in ("deadline", "pairs", "actions")
        and observation["actions"] <= 3
        and match_delivered(case, observation["delivered"])
        == set(case["expected_delivered_support_ids"])
        and not citation_violations(
            case, observation["citations"], observation["delivered"]
        )
    )


def join_arms(arms):
    if set(arms) != set(MODES):
        raise ValueError("all four arms required")
    indexed = {}
    for mode, rows in arms.items():
        if any(
            r["mode"] != mode
            or not all(
                r["snapshot"].get(k)
                for k in ("source", "answer", "reranker", "hardware")
            )
            for r in rows
        ):
            raise ValueError("missing/mislabelled snapshot")
        indexed[mode] = {r["case_id"]: r for r in rows}
        if len(indexed[mode]) != len(rows):
            raise ValueError("duplicate case observation")
    baseline = indexed["baseline"]
    if not baseline or any(set(rows) != set(baseline) for rows in indexed.values()):
        raise ValueError("missing paired cases")
    result = []
    for key, base in sorted(baseline.items()):
        group = {m: indexed[m][key] for m in MODES}
        if any(
            r["snapshot"] != base["snapshot"] or r["split"] != base["split"]
            for r in group.values()
        ):
            raise ValueError("source/model/hardware snapshot drift")
        result.append(group)
    return result


def evaluate(case, observation, review=None):
    from .evidence_quality_safety import actual_safety

    delivered = observation.get("delivered", [])
    matched = match_delivered(case, delivered)
    required = required_support(case)
    review = (
        review
        if review
        and review.get("answer_sha256") == text_digest(observation.get("answer", ""))
        else None
    )
    result = {
        **observation,
        "case_id": case["case_id"],
        "split": case["split"],
        "depth": case["depth"],
        "scenario": case["scenario"],
        "quality_aggregation": case["quality_aggregation"],
        "support_recall": support_recall(required, matched),
        "delivered_support_ids": sorted(matched),
        "packet_oracle_recall": support_recall(
            case["expected_delivered_support_ids"], matched
        ),
        "missing_upstream": sorted(
            required - match_delivered(case, observation.get("upstream", []))
        ),
        "lost_before_sdk": sorted(
            (required & match_delivered(case, observation.get("upstream", [])))
            - matched
        ),
        "citation_violations": citation_violations(
            case, observation.get("citations", []), delivered, frozen=False
        ),
        "authorization_violations": [
            s
            for s in delivered
            if identity(s)
            not in {identity(x) for x in case["sources"] if x["authorized_at_answer"]}
        ],
        "ndcg": ndcg(observation.get("ranked_ids", []), case.get("relevance_grades")),
        "safety": score_safety(case, observation),
        **review_claims(case, review),
    }
    result["frozen_v5_oracle"] = {
        "safety": result["safety"],
        "action_oracle": case.get("action_oracle"),
        "expected_delivered_support_ids": case["expected_delivered_support_ids"],
        "delivered_support_ids": sorted(matched),
        "citation_violations": citation_violations(
            case, observation.get("citations", []), delivered
        ),
    }
    result["actual_safety"] = actual_safety(result)
    return result
