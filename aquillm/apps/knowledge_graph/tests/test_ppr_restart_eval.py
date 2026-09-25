"""Offline PPR controls use authorized snapshots and the serving recurrence."""

import copy
import json

import pytest

from apps.knowledge_graph.evals.ppr_restart_fixtures import load_cases, parse_case
from apps.knowledge_graph.evals.run_ppr_restart_eval import evaluate_case, main


def fixture():
    return {
        "id": "two-node",
        "split": "regression",
        "question": "Who is A?",
        "nodes": ["A", "B"],
        "edges": [["A", "B", 1.0]],
        "seeds": [["A", 1.0]],
        "support_status": "supported",
        "support_summary": {"minimum_extraction_score": 0.9},
        "required_chunks": [1],
        "required_groups": [[1]],
    }


def test_two_node_controls_and_eight_step_reference():
    case = parse_case(fixture())
    fixed = evaluate_case(case, "fixed_020")
    adaptive = evaluate_case(case, "adaptive_v1")
    assert fixed["one_step_scores"] == pytest.approx({"A": 0.2, "B": 0.8})
    assert adaptive["one_step_scores"] == pytest.approx({"A": 0.35, "B": 0.65})
    assert fixed["effective_iterations"] == adaptive["effective_iterations"] == 8
    assert fixed["snapshot_checksum"] == adaptive["snapshot_checksum"]
    assert fixed["algorithm_signature"] != adaptive["algorithm_signature"]
    assert adaptive["reference"]["stopping_reason"] == "tolerance"
    assert adaptive["reference"]["residual_l1"] <= 1e-10
    assert fixed["metrics"]["candidate_recall"] == 1


def test_absolute_support_remains_part_of_execution_identity():
    strong = fixture()
    weak = copy.deepcopy(strong)
    weak["support_status"] = "insufficient"
    weak["support_summary"]["minimum_extraction_score"] = 0.3
    first = evaluate_case(parse_case(strong), "adaptive_v1")
    second = evaluate_case(parse_case(weak), "adaptive_v1")
    assert first["seed_checksum"] == second["seed_checksum"]
    assert first["execution_signature"] != second["execution_signature"]
    assert (first["effective_restart"], second["effective_restart"]) == (0.35, 0.2)


@pytest.mark.parametrize("mutation", ["duplicate", "negative", "nan", "outside"])
def test_rejects_invalid_snapshots(mutation):
    raw = fixture()
    if mutation == "duplicate":
        raw["nodes"].append("A")
    elif mutation == "negative":
        raw["edges"][0][2] = -1.0
    elif mutation == "nan":
        raw["edges"][0][2] = float("nan")
    else:
        raw["evidence_documents"] = ["outside"]
    with pytest.raises(ValueError):
        parse_case(raw)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        evaluate_case(parse_case(fixture()), "made_up")


def test_cli_regressions_do_not_claim_held_out_quality(tmp_path):
    output = tmp_path / "ppr.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["cases"]) >= 8
    assert report["quality_gate_status"] == "unmeasured_real_corpus"
    assert report["best_fixed_policy"] is None
    assert {row["support_status"] for row in report["cases"]} >= {
        "supported",
        "insufficient",
        "unknown",
    }
    assert report["factorial_replay"]["status"] == "synthetic_evidence_replay"


def test_loader_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "cases": [fixture(), fixture()]}))
    with pytest.raises(ValueError):
        load_cases(path)


def test_factorial_replay_exercises_graph_only_evidence_and_frozen_control(tmp_path):
    output = tmp_path / "factorial.json"
    main(["--output", str(output), "--fixed-choice", "fixed_035"])
    report = json.loads(output.read_text(encoding="utf-8"))
    arms = report["factorial_replay"]["arms"]
    selected = {
        arm: next(
            row["selected_chunk_ids"]
            for row in value["cases"]
            if row["id"] == "graph-sensitive"
        )
        for arm, value in arms.items()
    }
    assert selected["current"] == [7, 5]
    assert selected["A_and_B"] == [1, 7]
    assert selected["B_only"] != selected["current"]
    assert report["best_fixed_factorial_replay"]["baseline_policy"] == "fixed_035"
    assert report["factorial_replay"]["baseline_policy"] == "fixed_020"
    assert "seed_count" in report["quality_strata"]
    assert "policy_reason" in report["quality_strata"]
