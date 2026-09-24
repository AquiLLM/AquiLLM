"""Live scenario proof cannot be replaced by fixture flags or source/action stops."""

from copy import deepcopy

from apps.chat.evals.evidence_quality_safety import actual_safety, exhaustion


def observation():
    state = {
        "actions": 1,
        "sources": 2,
        "pairs": {"acquisition": 90, "final": 0},
        "text": {"materialized": 100, "tokenized": 200},
        "planner_calls": 0,
        "inflight": 0,
        "closed": None,
        "terminal": None,
        "remaining_ms": 4000,
    }
    limits = {
        "actions": 3,
        "unique_sources": 45,
        "acquisition_pairs": 90,
        "final_pairs": 45,
        "materialized_codepoints": 250000,
        "tokenized_codepoints": 1000000,
        "planner_calls": 2,
        "in_flight_pairs": 6,
        "retrieval_ms": 15000,
    }
    return {
        "publication_observation_complete": True,
        "late_publications": 0,
        "provenance_complete": True,
        "dispatch_accounting_complete": True,
        "source_validation_complete": True,
        "events": [
            {"event": "ledger_start", "ledger_id": "1", "limits": limits},
            {
                "event": "ledger",
                "ledger_id": "1",
                "operation": "reserve_pairs",
                "arguments": ["1"],
                "keywords": {"phase": "acquisition"},
                "before": state,
                "after": state,
                "result": False,
            },
            {"event": "turn_complete"},
        ],
    }


def test_pair_denial_preserves_final_reserve_and_requires_actual_charges():
    row = observation()
    assert exhaustion(row) == {"pairs": [1], "deadline": []}
    assert actual_safety(row)["passed"] is True
    changed = deepcopy(row)
    changed["events"][1]["before"]["pairs"]["acquisition"] = 0
    assert exhaustion(changed)["pairs"] == []
    changed["events"][1]["operation"] = "reserve_action"
    assert exhaustion(changed) == {"pairs": [], "deadline": []}


def test_missing_instrumentation_is_unknown_and_late_dispatch_is_failure():
    assert actual_safety({})["passed"] is None
    row = observation()
    row["events"][1]["after"]["terminal"] = "cancelled"
    row["events"].insert(2, {"event": "sdk_start"})
    assert actual_safety(row)["passed"] is False


def test_frozen_operational_adapter_validates_ideal_spans_and_fixture_is_ineligible(
    tmp_path,
):
    import json

    from apps.chat.evals.evidence_operational import (
        expand,
        load_workload,
        validate_attachment,
    )
    from apps.chat.evals.run_evidence_operational_eval import main

    data = load_workload()
    assert len(data["schedule"]["planned_run_ids"]) == 30
    pressure = expand(data["workloads"][-1], data)
    assert not pressure["required_claims"] and len(pressure["gold_support"]) == 24
    output = tmp_path / "operational.json"
    assert (
        main(
            [
                "--report",
                str(output),
                "--mode",
                "combined",
                "--profile",
                "pilot",
                "--repetition",
                "1",
                "--backend",
                "fixture",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["observations"]) == 4
    assert all(not r["operational"]["limit_pass"] for r in report["observations"])
    assert not validate_attachment([report], {})["passed"]


def test_operational_partial_requires_human_facts_and_real_pair_transport():
    from apps.chat.evals.evidence_operational import assess, load_workload
    from apps.chat.evals.evidence_quality_eval import text_digest

    data = load_workload()
    row = {**observation(), "answer": "Limited", "delivered": [], "citations": []}
    review = {
        "kind": "human",
        "reviewer": "reviewer",
        "answer_sha256": text_digest("Limited"),
        "bounded_partial": {
            k: True
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
        },
    }
    review["bounded_partial"].update(
        usable_fact_count=0,
        retained_fact_count=0,
        empty_reason="All factual windows rejected within actual limit",
    )
    assert not assess(data["workloads"][-1], data, row, review)["operational"][
        "limit_pass"
    ]
    row["events"].insert(1, {"event": "rerank_http", "pairs": 90})
    assert assess(data["workloads"][-1], data, row, review)["operational"]["limit_pass"]
    review["bounded_partial"]["usable_fact_count"] = 1
    assert not assess(data["workloads"][-1], data, row, review)["operational"][
        "limit_pass"
    ]
    row = observation()
    row["events"][1]["after"]["pairs"]["acquisition"] = 91
    assert actual_safety(row)["passed"] is False
