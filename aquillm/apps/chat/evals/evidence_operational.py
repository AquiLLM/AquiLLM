"""Frozen operational adapter and evidence predicates, outside quality aggregates."""

import hashlib
import json
from pathlib import Path

from .evidence_quality_eval import digest, evaluate, match_delivered
from .evidence_quality_safety import actual_safety, exhaustion

WORKLOAD_SHA256 = "ae60c47e40d7fa3574d6e8fa6c32e40eba51c30121e54741316d50314f8d3136"


def load_workload(path=None):
    path = path or Path(__file__).with_name("evidence_operational_cases.json")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKLOAD_SHA256:
        raise ValueError("operational workload changed")
    data = json.loads(raw)
    sources = {s["source_id"]: s for s in data["sources"]}
    for workload in data["workloads"]:
        for span in workload["required_support"] + workload.get("ideal_support", []):
            source = sources[span["source_id"]]
            if (
                span["source_id"] not in workload["source_ids"]
                or span["revision"] != source["revision"]
                or source["text"][span["start"] : span["end"]] != span["quote"]
            ):
                raise ValueError("invalid operational span")
    return data


def expand(workload, data):
    sources = [s for s in data["sources"] if s["source_id"] in workload["source_ids"]]
    citations = [{k: s[k] for k in ("source_id", "revision")} for s in sources]
    return {
        "case_id": workload["workload_id"],
        "split": "development",
        "depth": "deeper",
        "scenario": "operational",
        "quality_aggregation": "safety_only",
        "question": workload["question"],
        "turns": workload["turns"],
        "sources": sources,
        "gold_support": workload["required_support"]
        + workload.get("ideal_support", []),
        "required_claims": [
            {
                "claim_id": g["support_id"],
                "support_ids": [g["support_id"]],
                "labels": ["units", "conditions", "date"],
            }
            for g in workload["required_support"]
        ],
        "permitted_citations": citations,
        "publishable_citations": citations,
        "expected_delivered_support_ids": [],
        "forbidden_citations": [],
    }


def assess(workload, data, row, review=None):
    case = expand(workload, data)
    result = evaluate(case, row, review)
    safety = actual_safety(result)
    human = result["human_review"] if result["review_valid"] else {}
    events = row.get("events", [])
    trigger = exhaustion(row)
    first = {s[0] for s in (row.get("acquisition_source_rounds") or [[]])[0]}
    targets = {
        g["source_id"]
        for g in case["gold_support"]
        if g["support_id"] in workload.get("refinement_target_support_ids", [])
    }
    required = {g["support_id"] for g in workload["required_support"]}
    matched = match_delivered(case, row.get("delivered", []))
    actions = [i for i, e in enumerate(events) if e["event"] == "action_end"]
    planners = [
        i
        for i, e in enumerate(events)
        if e["event"] == "coverage_assessment"
        and e.get("unresolved_aspects")
        and e.get("next_action")
    ]
    refinement = bool(
        targets
        and not targets <= first
        and len(actions) > 1
        and planners
        and required <= matched
        and result["faithfulness"] == 1
        and result["qualification_accuracy"] == 1
        and safety["passed"] is True
    )
    checks = human.get("bounded_partial", {})
    from .evidence_bounded_partial import bounded_partial

    partial = bounded_partial(case, row, checks)
    charged = sum(e.get("pairs", 0) for e in events if e["event"] == "rerank_http")
    pair_limits = [
        e["limits"]["acquisition_pairs"] for e in events if e["event"] == "ledger_start"
    ]
    pair_proven = bool(trigger["pairs"] and pair_limits and charged >= pair_limits[0])
    limited = bool(
        (pair_proven or trigger["deadline"]) and partial and safety["passed"] is True
    )
    return {
        **result,
        "trace_sha256": digest(events),
        "operational": {
            "refinement_pass": refinement,
            "limit_pass": limited,
            "trigger": trigger,
            "pair_dispatch_proven": pair_proven,
            "partial_review_pass": partial,
            "first_pass_sufficient": bool(targets and targets <= first),
            "action_event_refs": actions,
            "planner_event_refs": planners,
            "ideal_support_recall": len(matched) / len(case["gold_support"])
            if case["gold_support"]
            else None,
        },
    }


def validate_attachment(reports, quality_reports):
    from .evidence_quality_review import verified_flags

    data = load_workload()
    expected = set(data["schedule"]["planned_run_ids"])
    rows = [r for report in reports for r in report.get("observations", [])]
    reasons = []
    if len(rows) != len(expected) or {r.get("run_id") for r in rows} != expected:
        reasons.append("missing or duplicate operational run IDs")
    workloads = {w["workload_id"]: w for w in data["workloads"]}
    covered = {
        m: {"refinement": False, "pair_or_deadline": False}
        for m in ("preservation", "combined")
    }
    for report in reports:
        if (
            report.get("backend") != "live"
            or verified_flags(report).get("runtime_verified") is not True
            or report.get("dirty") is not False
            or report.get("corpus_sha256") != WORKLOAD_SHA256
        ):
            reasons.append("unverified operational backend/runtime/workload")
        if (
            report.get("planned_run_ids") != data["schedule"]["planned_run_ids"]
            or not report.get("manifest_sha256")
            or report.get("concurrency") != 1
            or report.get("pair_capability_verified") is not True
        ):
            reasons.append("missing operational schedule/manifest/worker evidence")
        for row in report.get("observations", []):
            mode = row.get("mode")
            reference = quality_reports.get(mode, {})
            quality = reference.get("observations", [{}])[0]
            if (
                row.get("code_revision") != report.get("revision")
                or report.get("revision") != reference.get("revision")
                or any(
                    not row.get("snapshot", {}).get(k)
                    or row["snapshot"][k] != quality.get("snapshot", {}).get(k)
                    for k in ("answer", "reranker", "hardware", "embedding")
                )
            ):
                reasons.append("operational runtime/revision drift")
            workload = workloads.get(row.get("case_id"))
            if not workload or mode not in covered:
                reasons.append("unknown operational workload/mode")
                continue
            repetition = report.get("repetition", 0)
            run_id = f"ops-v2/{workload['workload_id']}/{mode}/r{repetition:02}"
            if (
                row.get("run_id") != run_id
                or report.get("mode") != mode
                or report.get("profile") != workload["profile"]
                or row.get("profile") != workload["profile"]
                or row.get("repetition") != repetition
            ):
                reasons.append("mislabelled operational run")
            controls = dict(row.get("comparison_controls", {}))
            reference_controls = dict(quality.get("comparison_controls", {}))
            if workload["profile"] == "one_action":
                reference_controls["max_actions"] = 1
            if not controls or controls != reference_controls:
                reasons.append("operational configuration drift")
            if row.get("snapshot", {}).get("source") != digest(
                expand(workload, data)["sources"]
            ):
                reasons.append("operational source drift")
            if row.get("trace_sha256") != digest(row.get("events", [])):
                reasons.append("operational trace digest mismatch")
            checked = assess(workload, data, row, row.get("human_review"))
            if checked["actual_safety"]["passed"] is not True:
                reasons.append("operational safety failed or unknown")
            if row.get("operational") != checked["operational"]:
                reasons.append("unsubstantiated operational assertions")
            proof = checked["operational"]
            covered[mode]["refinement"] |= proof["refinement_pass"]
            covered[mode]["pair_or_deadline"] |= (
                workload["profile"] == "pilot" and proof["limit_pass"]
            )
    for mode, classes in covered.items():
        for name, proven in classes.items():
            if not proven:
                reasons.append(f"{mode}: live {name} unproven")
    return {
        "passed": not reasons,
        "blocking_reasons": sorted(set(reasons)),
        "coverage": covered,
        "workload_sha256": WORKLOAD_SHA256,
        "report_digests": [digest(r) for r in reports],
    }
