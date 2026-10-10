"""Offline review packets preserve observations and require explicit human work."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from apps.chat.evals.evidence_quality_eval import digest
from apps.chat.tests.test_evidence_review_subject import reviewed_case


def sample():
    case, row, _ = reviewed_case()
    case = deepcopy(case)
    row = deepcopy(row)
    case["question"] = "¿Qué cambió en β? 🧪"
    case["turns"] = [{"role": "user", "content": "Résumé naïve"}]
    row["answer"] = "Señal β: 4 μg."
    row["delivered"] = [{"source_id": "synthetic", "start": 0, "end": 2, "text": "β🙂"}]
    row["citations"] = [{"source_id": "synthetic", "revision": "r"}]
    row["source_bindings"] = [{"span": "β🙂", "start": 0, "end": 2}]
    row["sdk_payloads"] = [{"text": "β🙂", "parts": [{"text": "Señal"}]}]
    row["events"] = [{"event": "final_sdk", "text": "Señal β"}]
    row["snapshot"]["source"] = digest(case["sources"])
    report = {
        "schema_version": 1,
        "backend": "live",
        "mode": "combined",
        "revision": "r",
        "observations": [row],
        "activation_eligible": False,
        "summary": {"pass": True, "score": 1},
    }
    return case, report


def completed(packet):
    responses = deepcopy(packet["response_template"])
    for response in responses["reviews"].values():
        response["reviewer"] = "Alex Reviewer"
        response["audit_attested"] = True
        response["answer_faithful"] = False
        response["citation_entailment"] = False
        for labels in response["claims"].values():
            for label in labels:
                labels[label] = False
    return responses


def test_export_preserves_exact_unicode_and_starts_all_human_fields_null():
    from apps.chat.evals.evidence_review_packet import prepare_packet

    case, report = sample()
    original = deepcopy(report)
    packet, manifest = prepare_packet(
        {"arm-A": report}, {case["case_id"]: case}, random_seed=9
    )
    assert report == original
    assert len(packet["inventory"]) == 1
    item = packet["items"][0]
    assert item["question"] == case["question"]
    assert item["turns"] == case["turns"]
    for field in (
        "answer",
        "citations",
        "delivered",
        "source_bindings",
        "sdk_payloads",
        "events",
    ):
        assert item[field] == report["observations"][0][field]
    assert packet["audit_supplement"][item["id"]] == report["observations"][0]
    assert "¿Qué cambió en β? 🧪" in packet["sheet"]
    assert "Résumé naïve" in packet["sheet"]
    assert "Señal β: 4 μg." in packet["sheet"]
    for forbidden in (
        "combined",
        "arm-A",
        "mode",
        "score",
        "pass",
        "activation_eligible",
    ):
        assert forbidden not in packet["sheet"]
    response = packet["response_template"]["reviews"][item["id"]]
    assert response["reviewer"] is None
    assert response["audit_attested"] is None
    assert response["answer_faithful"] is None
    assert response["citation_entailment"] is None
    assert all(
        value is None
        for labels in response["claims"].values()
        for value in labels.values()
    )
    assert manifest["reports"]["arm-A"] == digest(report)
    assert manifest["bindings"][item["id"]]["subject"] == response["subject"]


def test_export_inventories_invalid_safety_and_operational_rows_without_review_slots():
    from apps.chat.evals.evidence_review_packet import prepare_packet

    case, report = sample()
    invalid = deepcopy(report["observations"][0])
    invalid["invocation_id"] = ""
    invalid["case_id"] = "invalid"
    safety = deepcopy(report["observations"][0])
    safety["case_id"] = "safety"
    safety["quality_aggregation"] = "safety_only"
    operational = deepcopy(report["observations"][0])
    operational["case_id"] = "operational"
    operational["profile"] = "pilot"
    report["observations"].extend([invalid, safety, operational])
    cases = {
        case["case_id"]: case,
        "safety": {**case, "case_id": "safety", "quality_aggregation": "safety_only"},
        "operational": {**case, "case_id": "operational"},
    }
    packet, _ = prepare_packet({"r": report}, cases, random_seed=2)
    assert len(packet["inventory"]) == 4
    assert {entry["status"] for entry in packet["inventory"]} == {
        "reviewable",
        "invalid_subject",
        "separate_workflow",
    }
    assert len(packet["response_template"]["reviews"]) == 1


def test_opaque_ids_shuffle_report_order_deterministically():
    from apps.chat.evals.evidence_review_packet import prepare_packet

    case, report = sample()
    reports = {f"source-{number}": deepcopy(report) for number in range(5)}
    first, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=7)
    again, again_manifest = prepare_packet(
        reports, {case["case_id"]: case}, random_seed=7
    )
    other, other_manifest = prepare_packet(
        reports, {case["case_id"]: case}, random_seed=11
    )
    assert first == again and manifest == again_manifest
    assert [
        manifest["bindings"][item["id"]]["report_id"] for item in first["items"]
    ] != list(reports)
    assert [
        manifest["bindings"][item["id"]]["report_id"] for item in first["items"]
    ] != [
        other_manifest["bindings"][item["id"]]["report_id"] for item in other["items"]
    ]
    assert all(
        item["id"].startswith("R") and "source" not in item["id"]
        for item in first["items"]
    )


def test_fixture_report_cannot_create_review_slots_from_live_shaped_row():
    from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet

    case, report = sample()
    report["backend"] = "fixture"
    packet, manifest = prepare_packet(
        {"fixture": report}, {case["case_id"]: case}, random_seed=4
    )
    assert packet["inventory"][0]["status"] != "reviewable"
    assert packet["response_template"]["reviews"] == {}
    assert import_reviews(
        packet, manifest, {"fixture": report}, packet["response_template"]
    ) == {"fixture": {}}
    assert report["activation_eligible"] is False


@pytest.mark.parametrize(
    "changed", ["invocation_id", "sdk_payloads", "delivered", "events", "snapshot"]
)
def test_import_rejects_changed_original_evidence(changed):
    from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet

    case, report = sample()
    reports = {"r": report}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=1)
    altered = deepcopy(reports)
    row = altered["r"]["observations"][0]
    if changed == "snapshot":
        row[changed]["answer"] = "other"
    elif changed == "invocation_id":
        row[changed] = "other"
    else:
        row[changed].append({"changed": True})
    with pytest.raises(ValueError, match="report digest|observation|subject"):
        import_reviews(packet, manifest, altered, completed(packet))


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown",
        "packet_hash",
        "answer_hash",
        "subject",
        "reviewer",
        "audit",
        "null",
        "missing_label",
    ],
)
def test_import_rejects_stale_or_incomplete_responses(mutation):
    from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet

    case, report = sample()
    reports = {"r": report}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=1)
    responses = completed(packet)
    item_id = next(iter(responses["reviews"]))
    row = responses["reviews"][item_id]
    if mutation == "duplicate":
        responses["reviews"] = [{"id": item_id, **row}, {"id": item_id, **row}]
    elif mutation == "unknown":
        responses["reviews"]["unknown"] = deepcopy(row)
    elif mutation == "packet_hash":
        responses["packet_sha256"] = "changed"
    elif mutation == "answer_hash":
        row["answer_sha256"] = "changed"
    elif mutation == "subject":
        row["subject"]["observation_sha256"] = "changed"
    elif mutation == "reviewer":
        row["reviewer"] = ""
    elif mutation == "audit":
        row["audit_attested"] = None
    elif mutation == "null":
        row["answer_faithful"] = None
    elif mutation == "missing_label":
        labels = next(iter(row["claims"].values()))
        labels.pop(next(iter(labels)))
    with pytest.raises(ValueError):
        import_reviews(packet, manifest, reports, responses)


def test_import_rejects_changed_binding_status_and_audit_supplement():
    from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet

    case, report = sample()
    reports = {"r": report}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=1)
    item_id = packet["items"][0]["id"]
    changed_manifest = deepcopy(manifest)
    changed_manifest["bindings"][item_id]["status"] = "separate_workflow"
    with pytest.raises(ValueError, match="status|inventory"):
        import_reviews(packet, changed_manifest, reports, completed(packet))
    changed_packet = deepcopy(packet)
    changed_packet["audit_supplement"][item_id]["events"].append({"changed": True})
    with pytest.raises(ValueError, match="packet digest"):
        import_reviews(changed_packet, manifest, reports, completed(packet))


def test_import_keeps_false_labels_and_per_report_case_ids_separate():
    from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet

    case, report = sample()
    second = deepcopy(report)
    second["mode"] = "baseline"
    second["observations"][0]["mode"] = "baseline"
    from apps.chat.evals.evidence_effective_config import expected_treatment

    second["observations"][0]["resolved_treatment"] = expected_treatment("baseline")
    reports = {"first": report, "second": second}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=3)
    reviews = import_reviews(packet, manifest, reports, completed(packet))
    assert set(reviews) == {"first", "second"}
    for report_reviews in reviews.values():
        review = report_reviews[case["case_id"]]
        assert review["kind"] == "human"
        assert review["reviewer"] == "Alex Reviewer"
        assert review["answer_faithful"] is False
        assert review["citation_entailment"] is False
        assert all(
            v is False for labels in review["claims"].values() for v in labels.values()
        )
    assert report["activation_eligible"] is False
    assert second["activation_eligible"] is False


def test_offline_cli_roundtrip_rejects_unchanged_template(tmp_path):
    case, report = sample()
    runner = Path(__file__).parents[1] / "evals/run_evidence_review_packet.py"
    report_file = tmp_path / "observations.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "private"
    env = dict(os.environ)
    for key in list(env):
        if key == "SECRET_KEY" or key.startswith(
            ("DJANGO_", "POSTGRES_", "OPENAI_", "GEMINI_", "ANTHROPIC_", "GOOGLE_")
        ):
            env.pop(key)
    export = subprocess.run(
        [
            sys.executable,
            str(runner),
            "export",
            "--observations",
            str(report_file),
            "--output-dir",
            str(output),
            "--seed",
            "4",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert export.returncode == 0, export.stderr
    packet = json.loads((output / "packet.json").read_text(encoding="utf-8"))
    assert packet["audit_supplement"]
    imported = subprocess.run(
        [
            sys.executable,
            str(runner),
            "import",
            "--packet",
            str(output / "packet.json"),
            "--bindings",
            str(output / "bindings.json"),
            "--responses",
            str(output / "responses.json"),
            "--observations",
            str(report_file),
            "--output-dir",
            str(output / "imported"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert imported.returncode != 0
    assert not (output / "imported" / "report-001-reviews.json").exists()
    assert case["case_id"] == report["observations"][0]["case_id"]

    (output / "responses.json").write_text(
        json.dumps(completed(packet), ensure_ascii=False), encoding="utf-8"
    )
    accepted = subprocess.run(
        [
            sys.executable,
            str(runner),
            "import",
            "--packet",
            str(output / "packet.json"),
            "--bindings",
            str(output / "bindings.json"),
            "--responses",
            str(output / "responses.json"),
            "--observations",
            str(report_file),
            "--output-dir",
            str(output / "imported"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert accepted.returncode == 0, accepted.stderr
    imported_reviews = json.loads(
        (output / "imported" / "report-001-reviews.json").read_text(encoding="utf-8")
    )
    assert imported_reviews[case["case_id"]]["answer_faithful"] is False
