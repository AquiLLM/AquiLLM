"""Fail-closed enrollment of independently accepted semantic decision policies.

The trusted manifest is administrator-controlled configuration, never candidate
JSON. Without administrator enrollment no real human study is accepted.
Enrollment pins entire private artifacts, independent reviewer provenance,
predeclared criteria and both subject sets after out-of-band human review.
Removing a pin revokes the policy on the next load. No runtime activation occurs.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from .evidence_quality_eval import digest
from .evidence_review_subject import review_subject
from .evidence_support_capture import support_delivery_matches
from .evidence_support_metrics import (
    measure_cohorts,
    publication_matches,
    support_labels,
)
from .evidence_support_trust import trusted_artifact


@dataclass(frozen=True)
class ValidatedSupportPolicy:
    policy_id: str
    runtime_identity: str
    scope_kind: str
    decision_version: str
    review_artifact_digest: str
    valid_until: datetime


def reviewed_policy_record(attachment):
    """Read only a pinned artifact; hash agreement alone cannot enroll it."""
    record, trusted = trusted_artifact(attachment, "calibration")
    if (
        record["kind"] != "independent_support_policy"
        or record["reviewer"] != trusted["reviewer"]
        or record["provenance_sha256"] != trusted["provenance_sha256"]
        or digest(record["criteria"]) != trusted["criteria_sha256"]
        or digest(record["split_sha256"]) != trusted["subjects_sha256"]
    ):
        raise ValueError("support enrollment mismatch")
    return record


def validate_record(record, *, runtime_identity, scope_kind):
    for key in (
        "policy_id",
        "decision_version",
        "model_identity",
        "tokenizer_identity",
        "input_preparation_identity",
    ):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError("missing support policy identity")
    if (
        record["runtime_identity"] != runtime_identity
        or record["scope_kind"] != scope_kind
        or not isinstance(runtime_identity, str)
        or not runtime_identity.strip()
        or not isinstance(scope_kind, str)
        or not scope_kind.strip()
    ):
        raise ValueError("support runtime/scope mismatch")
    for key in ("corpus_sha256", "provenance_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", record[key]):
            raise ValueError("invalid frozen hash")
    expires = datetime.fromisoformat(record["valid_until"])
    declared = datetime.fromisoformat(record["predeclared_at"])
    opened = datetime.fromisoformat(record["heldout_opened_at"])
    now = datetime.now(UTC)
    if expires <= now or not declared < opened <= now:
        raise ValueError("expired or retrospective policy")
    rows = record["observations"]
    if not rows or len({(r["case_id"], r["mode"]) for r in rows}) != len(rows):
        raise ValueError("missing or repeated support samples")
    for row in rows:
        if (
            not publication_matches(row)
            or not support_delivery_matches(row)
            or support_labels(row) is None
            or row.get("runtime_identity") != runtime_identity
            or row.get("source_scope", {}).get("kind") != scope_kind
            or row.get("corpus_sha256") != record["corpus_sha256"]
            or row.get("policy_identity") != record["policy_id"]
            or not row.get("primary_question")
            or row.get("primary_question") != row.get("support_question")
            or row.get("citation_violations")
            or row.get("authorization_violations")
            or row.get("backend") != "live"
            or any(
                row.get(k) is not True
                for k in (
                    "provenance_complete",
                    "dispatch_accounting_complete",
                    "source_validation_complete",
                    "publication_observation_complete",
                )
            )
            or not isinstance(row.get("support_decision"), dict)
            or row["support_decision"].get("state")
            not in ("unknown", "insufficient", "partial", "complete")
        ):
            raise ValueError("invalid independently reviewed support observation")
    for split in ("development", "heldout"):
        subjects = [review_subject(r) for r in rows if r["split"] == split]
        if (
            not subjects
            or None in subjects
            or digest(subjects) != record["split_sha256"][split]
        ):
            raise ValueError("frozen support split drift")
    measure_cohorts(rows, record["criteria"])
    return expires


def load_support_policy(
    record: dict, *, runtime_identity: str, scope_kind: str
) -> ValidatedSupportPolicy | None:
    try:
        accepted = reviewed_policy_record(record)
        expires = validate_record(
            accepted, runtime_identity=runtime_identity, scope_kind=scope_kind
        )
        return ValidatedSupportPolicy(
            accepted["policy_id"],
            runtime_identity,
            scope_kind,
            accepted["decision_version"],
            record["artifact_sha256"],
            expires,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None
