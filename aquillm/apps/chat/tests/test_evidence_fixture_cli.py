"""Execute the public frozen CLI schedule; fixture output never measures quality."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def test_actual_fixture_cli_reports_preserve_frozen_cases_and_block_activation():
    root = Path(__file__).resolve().parents[4]
    evaluation = root / "aquillm/apps/chat/evals"
    destination = Path("/artifacts/r5-fixture-controls") / uuid4().hex
    destination.mkdir(parents=True)
    quality = evaluation / "evidence_quality_cases.json"
    operational = evaluation / "evidence_operational_cases.json"
    manifest = {
        "kind": "r5-fixture-cli-plumbing-v1",
        "quality_claim": "unmeasured",
        "backend": "fixture",
        "activation_eligible": False,
        "independent_human_reviews": False,
        "live_model_runs": 0,
        "public_corpora": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (quality, operational)
        },
        "commands": [],
    }
    failures = []

    def invoke(script, arguments, filename, expected_exit):
        report = destination / filename
        command = [
            sys.executable,
            str(evaluation / script),
            *arguments,
            "--report",
            str(report),
        ]
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=90)
        (destination / (filename + ".stdout")).write_bytes(completed.stdout)
        (destination / (filename + ".stderr")).write_bytes(completed.stderr)
        record = {
            "argv": command,
            "exit_code": completed.returncode,
            "expected_exit": expected_exit,
            "report": filename,
        }
        manifest["commands"].append(record)
        if completed.returncode != expected_exit or not report.exists():
            failures.append(record)
            return None
        record["sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        return json.loads(report.read_text())

    try:
        arms = ("baseline", "selection", "preservation", "combined")
        quality_rows = {}
        for split in ("development", "heldout"):
            for arm in arms:
                quality_rows[split, arm] = invoke(
                    "run_evidence_quality_eval.py",
                    ["--backend", "fixture", "--split", split, "--mode", arm],
                    f"quality-{split}-{arm}.json",
                    0,
                )
        workload = json.loads(operational.read_text())
        ops_paths, observed_ids = [], []
        for repetition in (1, 2, 3):
            order = (
                ("combined", "preservation")
                if repetition == 2
                else ("preservation", "combined")
            )
            for mode in order:
                for profile in ("pilot", "one_action"):
                    name = f"ops-{mode}-{profile}-r{repetition}.json"
                    row = invoke(
                        "run_evidence_operational_eval.py",
                        [
                            "--backend",
                            "fixture",
                            "--mode",
                            mode,
                            "--profile",
                            profile,
                            "--repetition",
                            str(repetition),
                        ],
                        name,
                        0,
                    )
                    ops_paths.append(str(destination / name))
                    if row:
                        assert row["backend"] == "fixture"
                        assert row["runtime_verified"] is False
                        assert row["activation_eligible"] is False
                        observed_ids.extend(r["run_id"] for r in row["observations"])
        comparisons = {}
        for split in ("development", "heldout"):
            comparisons[split] = invoke(
                "run_evidence_quality_eval.py",
                [
                    "--backend",
                    "fixture",
                    "--split",
                    split,
                    "--mode",
                    "combined",
                    "--compare",
                    *(str(destination / f"quality-{split}-{arm}.json") for arm in arms),
                    "--operational-reports",
                    *ops_paths,
                    "--require-activation",
                ],
                f"comparison-{split}.json",
                2,
            )
        assert not failures, failures
        for (split, arm), row in quality_rows.items():
            assert row["backend"] == "fixture" and row["quality_claim"] == "unmeasured"
            assert row["split"] == split and row["mode"] == arm
            assert row["activation_eligible"] is False
            assert len(row["observations"]) == 40
            assert all(r["faithfulness"] is None for r in row["observations"])
        for arm in arms:
            identifiers = [
                r["case_id"]
                for split in ("development", "heldout")
                for r in quality_rows[split, arm]["observations"]
            ]
            assert len(identifiers) == len(set(identifiers)) == 80
        assert len(observed_ids) == len(set(observed_ids)) == 30
        assert set(observed_ids) == set(workload["schedule"]["planned_run_ids"])
        for row in comparisons.values():
            assert row["activation_eligible"] is False
            assert row["gate"]["activation_eligible"] is False
            reasons = row["gate"]["blocking_reasons"]
            assert any("fixture is ineligible" in reason for reason in reasons)
            assert any("runtime_verified" in reason for reason in reasons)
            assert any("human" in reason for reason in reasons)
        from apps.chat.evals.retrieval_treatment_manifest import treatment_manifest

        report_bytes = {
            row["report"]: (destination / row["report"]).read_bytes()
            for row in manifest["commands"]
        }
        # Existing four-arm fixture plumbing is a bundled comparison, not an
        # independently executed window/source/capacity/followup treatment.
        catalog = treatment_manifest(
            configured={
                "window_source": {
                    "requested_bundle": "existing_activation_v2_four_arm",
                    "independent_window_source": False,
                }
            },
            reports=report_bytes,
            associations={"window_source": list(report_bytes)},
            dependencies={"I5": "pending_integration", "R4": "pending_review"},
        )
        assert all(
            row["result"] == "inconclusive" for row in catalog["treatments"].values()
        )
        for row in manifest["commands"]:
            assert catalog["reports"][row["report"]]["sha256"] == row["sha256"]
        catalog_bytes = (json.dumps(catalog, indent=2) + "\n").encode()
        (destination / "treatment-catalog.json").write_bytes(catalog_bytes)
        manifest["treatment_catalog_sha256"] = hashlib.sha256(catalog_bytes).hexdigest()
        manifest.update(
            quality_cases_per_arm=80,
            quality_arms=4,
            fixture_operational_records=30,
            measured_operational_runs=0,
            comparison_blocking_reasons={
                split: row["gate"]["blocking_reasons"]
                for split, row in comparisons.items()
            },
        )
    finally:
        manifest["failures"] = failures
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print("R5_FIXTURE_ARTIFACT=" + str(destination))
