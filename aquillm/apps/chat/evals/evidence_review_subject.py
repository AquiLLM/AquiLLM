"""Canonical human-review identity from original observations, stable on rescore."""

from pathlib import Path

from .evidence_effective_config import expected_treatment
from .evidence_observation_json import normalize
from .evidence_quality_eval import MODES, digest, load_cases, review_claims, text_digest

# Only observations belong here: never derived metrics, reviews or their digests.
ORIGINAL_FIELDS = (
    "case_id",
    "split",
    "mode",
    "backend",
    "invocation_id",
    "code_revision",
    "run_id",
    "repetition",
    "profile",
    "snapshot",
    "comparison_controls",
    "resolved_treatment",
    "answer",
    "delivered",
    "citations",
    "sdk_payloads",
    "events",
    "upstream",
    "acquisition_source_rounds",
    "timings_ms",
    "cache_state",
    "dispatches",
    "pairs",
    "coverage",
    "stop_reason",
    "actions",
    "closed",
    "late_publications",
    "published_after_disconnect",
    "observation_failed",
    "provenance_complete",
    "dispatch_accounting_complete",
    "publication_observation_complete",
    "source_validation_complete",
    "source_bindings",
    "error",
)


def review_subject(row):
    """Missing or malformed original identity cannot be reviewed as live evidence."""
    try:
        row = normalize(row)
        if (
            row.get("observation_normalization_failed")
            or row.get("mode") not in MODES
            or row.get("backend") != "live"
            or row.get("split") not in ("development", "heldout")
            or row.get("profile") not in ("quality", "pilot", "one_action")
            or row.get("resolved_treatment") != expected_treatment(row.get("mode"))
            or type(row.get("repetition")) is not int
            or row["repetition"] < 1
            or not all(
                isinstance(row.get(k), str) and row[k].strip()
                for k in ("case_id", "run_id", "invocation_id", "code_revision")
            )
            or not isinstance(row.get("answer"), str)
            or not all(
                isinstance(row.get(k), list)
                for k in ("delivered", "citations", "sdk_payloads", "events")
            )
            or not isinstance(row.get("comparison_controls"), dict)
            or not row["comparison_controls"]
            or not all(
                isinstance(row.get("snapshot", {}).get(k), str) and row["snapshot"][k]
                for k in (
                    "source",
                    "answer",
                    "reranker",
                    "hardware",
                    "embedding",
                    "configuration",
                    "bindings",
                )
            )
            or row["snapshot"]["configuration"] != digest(row["comparison_controls"])
        ):
            return None
        original = {key: row[key] for key in ORIGINAL_FIELDS if key in row}
        return {
            "schema": "evidence-review-subject-v1",
            "observation_sha256": digest(original),
            "answer_sha256": text_digest(row["answer"]),
        }
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def validated_review(row, review):
    subject = review_subject(row)
    return (
        review
        if (
            subject
            and isinstance(review, dict)
            and review.get("kind") == "human"
            and isinstance(review.get("reviewer"), str)
            and review["reviewer"].strip()
            and isinstance(review.get("claims", {}), dict)
            and all(isinstance(v, dict) for v in review.get("claims", {}).values())
            and isinstance(review.get("bounded_partial", {}), dict)
            and review.get("subject") == subject
            and review.get("answer_sha256") == subject["answer_sha256"]
        )
        else None
    )


def quality_review_errors(reports):
    cases = {
        c["case_id"]: c
        for c in load_cases(Path(__file__).with_name("evidence_quality_cases.json"))
    }
    errors = []
    for mode, report in reports.items():
        for row in report.get("observations", []):
            if row.get("quality_aggregation") != "normal":
                continue
            review = validated_review(row, row.get("human_review"))
            case = cases.get(row.get("case_id"))
            if (
                not review
                or not case
                or row.get("code_revision") != report.get("revision")
                or row.get("snapshot", {}).get("source") != digest(case["sources"])
                or any(row.get(k) != v for k, v in review_claims(case, review).items())
            ):
                errors.append(f"{mode}: missing/stale human observation review")
    return errors
