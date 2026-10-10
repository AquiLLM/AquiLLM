"""Current-run human acceptance, distinct from reusable policy calibration."""

from datetime import UTC, datetime

from .evidence_quality_eval import digest
from .evidence_review_subject import review_subject, validated_review
from .evidence_support_trust import trusted_artifact


def activation_matches(report, evidence, policy_record, policy):
    record, pin = trusted_artifact(evidence["support_activation"], "activation")
    rows = sorted(report["observations"], key=lambda r: r["case_id"])
    enrolled = sorted(record["observations"], key=lambda r: r["case_id"])
    subjects = [review_subject(r) for r in rows]
    enrolled_subjects = [review_subject(r) for r in enrolled]
    if (
        record["kind"] != "independent_support_activation"
        or record["revision"] != pin["revision"]
        or pin["revision"] != report["revision"]
        or record["policy_sha256"] != pin["policy_sha256"]
        or pin["policy_sha256"] != policy.review_artifact_digest
        or record["corpus_sha256"] != report["corpus_sha256"]
        or record["corpus_sha256"] != policy_record["corpus_sha256"]
        or digest(record["criteria"]) != pin["criteria_sha256"]
        or record["criteria"] != policy_record["criteria"]
        or record["subjects_sha256"] != pin["subjects_sha256"]
        or digest(subjects) != pin["subjects_sha256"]
        or subjects != enrolled_subjects
        or not subjects
        or None in subjects
        or len({r["case_id"] for r in rows}) != len(rows)
        or datetime.fromisoformat(record["valid_until"]) <= datetime.now(UTC)
    ):
        return False
    return all(
        row["code_revision"] == report["revision"]
        and row.get("runtime_identity") == policy.runtime_identity
        and row.get("source_scope", {}).get("kind") == policy.scope_kind
        and row.get("policy_identity") == policy.policy_id
        and validated_review(row, row.get("human_review")) is not None
        and row["human_review"] == accepted["human_review"]
        for row, accepted in zip(rows, enrolled, strict=True)
    )
