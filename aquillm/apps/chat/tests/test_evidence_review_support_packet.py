"""Human support judgments survive the real offline packet/import boundary."""

from copy import deepcopy

import pytest

from apps.chat.evals.evidence_review_packet import _sheet, import_reviews, prepare_packet
from apps.chat.evals.evidence_quality_eval import digest
from apps.chat.evals.evidence_support_metrics import support_labels
from apps.chat.tests.test_evidence_review_packet import completed, sample


LABELS = {
    "quote_accuracy": True,
    "source_entailment": False,
    "inference": True,
    "numerical_conditions": False,
    "contradiction_free": True,
    "coverage": False,
    "false_complete": False,
    "unsupported_answer": True,
    "abstained": False,
    "abstention_correct": False,
}


def support_packet():
    case, report = sample()
    row = report["observations"][0]
    row["support_decision"] = {"state": "unknown", "basis": "SUPPORT_STATE_CANARY"}
    row["primary_question"] = case["question"]
    row["support_question"] = case["question"]
    row["score_metadata"] = {"raw_score": "SCORE_CANARY"}
    row["policy_identity"] = "POLICY_CANARY"
    row["human_review"] = {"support_labels": {key: True for key in LABELS}}
    case["required_claims"][0]["statement"] = "GOLD_CLAIM_CANARY"
    reports = {"r": report}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=3)
    responses = completed(packet)
    key = packet["items"][0]["id"]
    return case, reports, packet, manifest, responses, key


def test_support_packet_starts_blank_without_gold_or_assessment_leakage():
    _, reports, packet, _, _, key = support_packet()
    template = packet["response_template"]["reviews"][key]
    assert template["support_labels"] == {label: None for label in LABELS}
    assert "Señal β: 4 μg." in packet["sheet"]
    assert "β🙂" in packet["sheet"]
    assert "¿Qué cambió en β? 🧪" in packet["sheet"]
    for secret in ("GOLD_CLAIM_CANARY", "SUPPORT_STATE_CANARY", "SCORE_CANARY", "POLICY_CANARY"):
        assert secret not in packet["sheet"]
    assert packet["audit_supplement"][key] == reports["r"]["observations"][0]


def test_import_preserves_mixed_human_support_labels_for_calibration():
    case, reports, packet, manifest, responses, key = support_packet()
    responses["reviews"][key]["support_labels"] = deepcopy(LABELS)
    original = deepcopy(responses)
    review = import_reviews(packet, manifest, reports, responses)["r"][case["case_id"]]
    row = {**reports["r"]["observations"][0], "human_review": review}
    assert support_labels(row) == LABELS
    assert responses == original
    assert reports["r"]["activation_eligible"] is False
    responses["reviews"][key]["support_labels"]["source_entailment"] = True
    assert review["support_labels"]["source_entailment"] is False


@pytest.mark.parametrize("mutation", ["absent", "missing", "extra", "none", "integer", "string", "list"])
def test_support_import_rejects_incomplete_or_nonboolean_judgments(mutation):
    _, reports, packet, manifest, responses, key = support_packet()
    response = responses["reviews"][key]
    response["support_labels"] = deepcopy(LABELS)
    if mutation == "absent":
        del response["support_labels"]
    elif mutation == "missing":
        del response["support_labels"]["coverage"]
    elif mutation == "extra":
        response["support_labels"]["model_approved"] = True
    elif mutation == "list":
        response["support_labels"] = list(LABELS)
    else:
        response["support_labels"]["coverage"] = {"none": None, "integer": 1, "string": "true"}[mutation]
    with pytest.raises(ValueError, match="support"):
        import_reviews(packet, manifest, reports, responses)


def test_legacy_packet_cannot_inject_support_judgments():
    case, report = sample()
    reports = {"r": report}
    packet, manifest = prepare_packet(reports, {case["case_id"]: case}, random_seed=3)
    responses = completed(packet)
    key = packet["items"][0]["id"]
    assert "support_labels" not in responses["reviews"][key]
    assert "support_labels" not in import_reviews(packet, manifest, reports, responses)["r"][case["case_id"]]
    responses["reviews"][key]["support_labels"] = deepcopy(LABELS)
    with pytest.raises(ValueError, match="support"):
        import_reviews(packet, manifest, reports, responses)


def test_changed_support_subject_cannot_reuse_completed_review():
    _, reports, packet, manifest, responses, key = support_packet()
    responses["reviews"][key]["support_labels"] = deepcopy(LABELS)
    reports["r"]["observations"][0]["support_decision"]["state"] = "complete"
    with pytest.raises(ValueError, match="digest|subject"):
        import_reviews(packet, manifest, reports, responses)


@pytest.mark.parametrize("fields", [
    ("primary_question",), ("support_question",), ("primary_question", "support_question"),
])
@pytest.mark.parametrize("value", [None, "A materially different actual question"])
def test_support_packet_rejects_unbound_display_question(fields, value):
    case, reports, *_ = support_packet()
    row = reports["r"]["observations"][0]
    for field in fields:
        if value is None:
            del row[field]
        else:
            row[field] = value
    packet, _ = prepare_packet(reports, {case["case_id"]: case}, random_seed=3)
    assert packet["inventory"] == [{"id": "R0001", "status": "invalid_case_binding"}]
    assert packet["items"] == [] and packet["response_template"]["reviews"] == {}


@pytest.mark.parametrize("changed", ["item", "sheet"])
def test_import_rejects_reprepared_packet_with_different_display_question(changed):
    _, reports, packet, manifest, responses, key = support_packet()
    if changed == "item":
        packet["items"][0]["question"] = "A materially different displayed question"
        packet["sheet"] = _sheet(packet["items"])
    else:
        packet["sheet"] = "A materially different question and answer"
    template = {**packet["response_template"], "packet_sha256": None}
    checksum = digest({**packet, "response_template": template})
    manifest["packet_sha256"] = checksum
    packet["response_template"]["packet_sha256"] = checksum
    responses["packet_sha256"] = checksum
    responses["reviews"][key]["support_labels"] = deepcopy(LABELS)
    with pytest.raises(ValueError, match="question"):
        import_reviews(packet, manifest, reports, responses)


@pytest.mark.parametrize("field", [
    "answer", "delivered", "citations", "source_bindings", "sdk_payloads", "events",
])
def test_import_rejects_reprepared_item_with_different_original_evidence(field):
    _, reports, packet, manifest, responses, key = support_packet()
    packet["items"][0][field] = "Different answer" if field == "answer" else ["Different evidence"]
    packet["sheet"] = _sheet(packet["items"])
    template = {**packet["response_template"], "packet_sha256": None}
    checksum = digest({**packet, "response_template": template})
    manifest["packet_sha256"] = checksum
    packet["response_template"]["packet_sha256"] = checksum
    responses["packet_sha256"] = checksum
    responses["reviews"][key]["support_labels"] = deepcopy(LABELS)
    with pytest.raises(ValueError, match="support"):
        import_reviews(packet, manifest, reports, responses)
