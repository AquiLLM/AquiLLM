"""Public regression supplement and actual offline runner boundaries."""

import importlib
import json
import subprocess
import sys

import pytest

from apps.knowledge_graph.evals import run_kg_eval


def test_supplement_is_separate_synthetic_and_has_all_graph_triggers():
    try:
        module = importlib.import_module(
            "apps.knowledge_graph.evals.graph_remediation_cases"
        )
    except ModuleNotFoundError:
        pytest.fail("G6 synthetic supplement loader missing")
    data = module.load_cases()
    assert data["adoption_eligible"] is False
    assert {r["lane"] for r in data["cases"]} == {"G1", "G2", "G3", "G4", "G5"}
    assert len(run_kg_eval.load_extraction_cases()) == 6
    assert len(run_kg_eval.load_retrieval_cases()) == 9
    for row in data["cases"]:
        assert "labels" not in module.execution_inputs(row)


def test_runner_records_absolute_identity_counts():
    case = run_kg_eval.load_extraction_cases()[0]
    score = run_kg_eval.score_extraction(case, case["expected"])
    assert "automatic_correct" in score
    assert score["automatic_correct"] == score["automatic_total"]
    assert score["automatic_incorrect"] == 0
    assert "candidate_correct" in score


def test_actual_graph_cli_persists_legacy_ineligible(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"auto_link_precision": 1}))
    output = tmp_path / "gates.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.knowledge_graph.evals.run_graph_quality_gates_v2",
            "--report",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert output.exists(), completed.stderr
    result = json.loads(output.read_text())
    assert completed.returncode == 2
    assert result["result"]["eligible"] is False
    assert "legacy_report_ineligible" in result["result"]["blocking_reasons"]
    print("G6 CLI persisted legacy report; exit=2; eligible=false")
