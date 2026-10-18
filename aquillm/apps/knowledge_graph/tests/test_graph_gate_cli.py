"""Actual credential-free CLI parsing and activation comparison boundaries."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apps.knowledge_graph.evals.run_graph_quality_gates_v2 import load_graph_report
from apps.knowledge_graph.tests.graph_gate_test_fixtures import fixture_report


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', "[]"])
def test_original_graph_artifact_parser_rejects_ambiguous_bytes(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_graph_report(path)


def test_actual_numerical_cli_has_no_activation_authority(tmp_path):
    source = tmp_path / "synthetic-positive.json"
    source.write_text(json.dumps(fixture_report()))
    output = tmp_path / "gate.json"
    command = [
        sys.executable,
        "-m",
        "apps.knowledge_graph.evals.run_graph_quality_gates_v2",
        "--report",
        str(source),
        "--output",
        str(output),
    ]
    done = subprocess.run(
        command, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(output.read_text())
    assert result["result"]["eligible"]
    assert "activation_eligible" not in result
    print("G6 numerical synthetic-injected CLI exit=0; no activation authority")


def test_existing_comparison_cli_keeps_fixture_and_missing_acceptance_off(tmp_path):
    root = Path(__file__).resolve().parents[3]
    cli = root / "apps/chat/evals/run_evidence_quality_eval.py"
    reports = []
    for mode in ("baseline", "selection", "preservation", "combined"):
        path = tmp_path / (mode + ".json")
        done = subprocess.run(
            [
                sys.executable,
                str(cli),
                "--backend",
                "fixture",
                "--mode",
                mode,
                "--split",
                "heldout",
                "--report",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        reports.append(str(path))
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(fixture_report()))
    output = tmp_path / "comparison.json"
    done = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--backend",
            "fixture",
            "--mode",
            "combined",
            "--split",
            "heldout",
            "--report",
            str(output),
            "--compare",
            *reports,
            "--graph-report",
            str(source),
            "--require-activation",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2, done.stderr
    result = json.loads(output.read_text())
    assert not result["activation_eligible"]
    assert result["gate"]["gate_version"] == "evidence-activation-v3-support"
    assert "graph_applicability_unknown" in result["gate"]["blocking_reasons"]
    assert result["gate"]["operational_evidence"]["passed"] is False
    print("G6 existing four-arm fixture comparison CLI exit=2; activation=false")
