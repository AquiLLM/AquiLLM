"""Real supported SDK values have the same review identity before/after JSON."""

import json
from dataclasses import dataclass
from uuid import uuid4

from google.genai.types import Content, GenerateContentConfig, Part

from apps.chat.evals.evidence_quality_eval import evaluate
from apps.chat.evals.evidence_review_subject import review_subject, validated_review
from apps.chat.tests.test_evidence_review_subject import reviewed_case


def test_typed_gemini_subject_is_stable_and_payload_changes_invalidate_review():
    case, row, review = reviewed_case()
    payload = {
        "contents": [Content(role="user", parts=[Part(text="Exact evidence")])],
        "config": GenerateContentConfig(max_output_tokens=42),
    }
    row["sdk_payloads"] = [payload]
    row["events"] = [{"event": "sdk_start", "payload": payload}]
    subject = review_subject(row)
    assert subject is not None
    review["subject"] = subject
    result = evaluate(case, row, review)
    # Actual report boundary uses the same shared normalization contract.
    from apps.chat.evals.evidence_observation_json import normalize

    saved = json.loads(json.dumps(normalize(result)))
    assert saved["review_subject"] == subject == review_subject(saved)
    assert evaluate(case, saved, saved["human_review"])["review_valid"]
    saved["sdk_payloads"][0]["contents"][0]["parts"][0]["text"] = "Changed"
    assert validated_review(saved, review) is None


def test_supported_dataclass_uuid_and_unsupported_values():
    from apps.chat.evals.evidence_observation_json import normalize

    @dataclass
    class Item:
        key: object

    key = uuid4()
    assert normalize({"item": Item(key), "tuple": (1, 2)}) == {
        "item": {"key": str(key)},
        "tuple": [1, 2],
    }
    case, row, _ = reviewed_case()
    row["sdk_payloads"] = [object()]
    assert review_subject(row) is None
    result = evaluate(case, row)
    assert result["review_subject"] is None and not result["provenance_complete"]
    assert json.loads(json.dumps(normalize(result)))["observation_normalization_failed"]


def test_partial_proof_compares_typed_and_saved_sdk_payloads():
    from apps.chat.evals.evidence_bounded_partial import bounded_partial
    from apps.chat.tests.test_evidence_bounded_partial import partial_case

    case, row, checks = partial_case()
    payload = {
        "contents": [
            Content(role="user", parts=[Part(text=row["sdk_payloads"][0]["content"])])
        ],
        "config": GenerateContentConfig(max_output_tokens=42),
    }
    row["sdk_payloads"] = [payload]
    row["events"][-1]["payload"] = json.loads(
        json.dumps(payload, default=lambda x: x.model_dump(mode="json"))
    )
    assert bounded_partial(case, row, checks)
