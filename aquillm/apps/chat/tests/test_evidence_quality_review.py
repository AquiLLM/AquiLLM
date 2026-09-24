"""Human/operator records must bind immutable answers, sources and run evidence."""

import json
from pathlib import Path

import pytest

from apps.chat.evals.evidence_quality_eval import (
    digest,
    evaluate,
    load_cases,
    text_digest,
)
from apps.chat.evals.evidence_quality_review import attach_evidence, load_observations


def test_stale_answer_review_and_model_self_review_never_count():
    case = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")[
        0
    ]
    observation = {"answer": "wrong unit", "delivered": [], "citations": []}
    review = {
        "reviewer": "person",
        "kind": "human",
        "answer_sha256": text_digest("different answer"),
        "answer_faithful": True,
        "citation_entailment": True,
        "claims": {
            c["claim_id"]: {k: True for k in ["faithful", *c["labels"]]}
            for c in case["required_claims"]
        },
    }
    assert evaluate(case, observation, review)["faithfulness"] is None
    review["answer_sha256"] = text_digest(observation["answer"])
    review["answer_faithful"] = False
    assert evaluate(case, observation, review)["faithfulness"] < 1
    review["kind"] = "model"
    assert evaluate(case, observation, review)["faithfulness"] is None


def test_rollout_artifact_hash_and_measured_reserve(tmp_path):
    artifact = tmp_path / "measurements.json"
    artifact.write_text(json.dumps({"measurements": []}))
    import hashlib

    record = {
        "reviewed_by": "operator",
        "status": "passed",
        "artifact": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "authorization_packet_p95_ms": 1001,
        "configured_reserve_ms": 1000,
    }
    report = {
        "revision": "r",
        "corpus_sha256": "c",
        "observations": [
            {
                "case_id": "1",
                "snapshot": {"source": "s"},
                "comparison_controls": {"completion_reserve_ms": 1000},
            }
        ],
    }
    evidence = {
        "kind": "human_operator",
        "reviewer": "operator",
        "revision": "r",
        "corpus_sha256": "c",
        "snapshot_digest": digest([{"source": "s"}]),
        "completion_reserve_measured": record,
    }
    assert not attach_evidence(report, evidence)["completion_reserve_measured"]
    record["authorization_packet_p95_ms"] = 999
    assert attach_evidence(report, evidence)["completion_reserve_measured"]
    record["configured_reserve_ms"] = 10000
    assert not attach_evidence(report, evidence)["completion_reserve_measured"]
    record["configured_reserve_ms"] = 1000
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        attach_evidence(report, evidence)


def test_reannotation_joins_case_id_and_rejects_source_drift():
    cases = load_cases(Path(__file__).parents[1] / "evals/evidence_quality_cases.json")[
        :2
    ]
    rows = [
        {
            "case_id": c["case_id"],
            "split": c["split"],
            "snapshot": {"source": digest(c["sources"])},
        }
        for c in cases
    ]
    report = {"backend": "live", "mode": "combined", "observations": rows[::-1]}
    assert load_observations(report, cases, backend="live", mode="combined") == rows
    rows[0]["snapshot"]["source"] = "different"
    with pytest.raises(ValueError, match="drift"):
        load_observations(report, cases, backend="live", mode="combined")


def test_standalone_fixture_cli_needs_no_django_credentials(tmp_path):
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    for key in list(environment):
        if key == "SECRET_KEY" or key.startswith(
            ("DJANGO_", "POSTGRES_", "OPENAI_", "GEMINI_", "ANTHROPIC_", "GOOGLE_")
        ):
            environment.pop(key)
    runner = Path(__file__).parents[1] / "evals/run_evidence_quality_eval.py"
    report = tmp_path / "fixture.json"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--backend",
            "fixture",
            "--split",
            "development",
            "--mode",
            "combined",
            "--report",
            str(report),
            "--require-activation",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["observations"]) == 40
    assert data["activation_eligible"] is False
