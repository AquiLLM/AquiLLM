"""Offline replay checks against actual selected evidence identities."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apps.chat.evals.evidence_selection_fixtures import load_cases
from apps.chat.evals.evidence_selection_metrics import paired_bootstrap_interval
from apps.chat.evals.run_evidence_selection_eval import evaluate_case, main


def fixture():
    return {
        "id": "complementary",
        "split": "regression",
        "family": "factual",
        "question": "What did the trial find?",
        "profile": "focused",
        "limits": {"max_passages": 2, "max_per_document": 3, "token_budget": 1000},
        "candidates": [
            {
                "chunk_id": 1,
                "doc_id": "a",
                "text": "Trial outcome was positive.",
                "relevance": 0.95,
                "fused_rank": 1,
            },
            {
                "chunk_id": 2,
                "doc_id": "a",
                "text": "Validation confirmed it.",
                "relevance": 0.92,
                "fused_rank": 2,
            },
            {
                "chunk_id": 3,
                "doc_id": "b",
                "text": "Unrelated context.",
                "relevance": 0.25,
                "fused_rank": 3,
            },
        ],
        "grades": {"1": 3, "2": 3, "3": 0},
        "required_chunks": [1, 2],
        "required_pairs": [[1, 2]],
        "forbidden_chunks": [3],
        "aspects": {"finding": [1], "validation": [2]},
        "expected_adaptive": [1, 2],
    }


def load_one(tmp_path, case):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": 1, "cases": [case]}), encoding="utf-8")
    return load_cases(path)[0]


def test_adaptive_keeps_complementary_evidence_and_legacy_displaces_it(tmp_path):
    case = load_one(tmp_path, fixture())
    adaptive = evaluate_case(case, "adaptive")
    legacy = evaluate_case(case, "legacy")
    assert adaptive["selected_chunk_ids"] == [1, 2]
    assert legacy["selected_chunk_ids"] == [1, 3]
    assert adaptive["metrics"]["required_pair_recall"] == 1
    assert legacy["metrics"]["required_pair_recall"] == 0
    assert adaptive["metrics"]["aspect_coverage"] == 1
    assert legacy["metrics"]["supporting_chunk_recall"] == 0.5
    assert adaptive["metrics"]["ndcg"] == 1
    assert adaptive["expectation_passed"] is True


def test_missing_upstream_gold_stays_a_miss(tmp_path):
    raw = fixture()
    raw["required_chunks"].append(99)
    raw["grades"]["99"] = 3
    case = load_one(tmp_path, raw)
    result = evaluate_case(case, "adaptive")
    assert result["metrics"]["supporting_chunk_recall"] == pytest.approx(2 / 3)
    assert result["missing_upstream_gold"] == [99]


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "limit", "expected"])
def test_rejects_invalid_replay_inputs(tmp_path, mutation):
    raw = fixture()
    if mutation == "duplicate":
        raw["candidates"].append(copy.deepcopy(raw["candidates"][0]))
    elif mutation == "nan":
        raw["candidates"][0]["relevance"] = float("nan")
    elif mutation == "limit":
        raw["limits"]["max_passages"] = -1
    else:
        raw["expected_adaptive"] = [99]
    with pytest.raises(ValueError):
        load_one(tmp_path, raw)


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        evaluate_case(load_one(tmp_path, fixture()), "invented")


def test_bootstrap_is_paired_deterministic_and_reports_sample_count():
    result = paired_bootstrap_interval([0.2, 0.4, 0.6], resamples=500)
    assert result == paired_bootstrap_interval([0.2, 0.4, 0.6], resamples=500)
    assert result["count"] == 3
    assert result["lower"] <= result["mean"] <= result["upper"]
    assert paired_bootstrap_interval([])["mean"] is None


def test_cli_runs_tracked_regressions_and_marks_quality_gates_unmeasured(tmp_path):
    output = tmp_path / "report.json"
    assert (
        main(["--policy", "adaptive", "--split", "regression", "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["cases"]) >= 9
    assert report["quality_gate_status"] == "unmeasured_real_corpus"
    assert report["comparison_to_legacy"]["ndcg"]["count"] >= 9
    assert report["families"]
    assert all(row["selected_citations"] for row in report["cases"])


def test_standalone_cli_needs_no_django_secrets_or_live_services(tmp_path):
    environment = dict(os.environ)
    environment.pop("SECRET_KEY", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.chat.evals.run_evidence_selection_eval",
            "--output",
            str(tmp_path / "standalone.json"),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
