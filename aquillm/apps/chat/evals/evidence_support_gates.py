"""Support evidence joins the existing four-arm gate; never a new authority."""

from .evidence_quality_eval import digest, text_digest
from .evidence_review_subject import review_subject
from .evidence_support_calibration import load_support_policy, reviewed_policy_record
from .evidence_support_capture import support_delivery_matches
from .evidence_support_metrics import (
    DIMENSIONS,
    measure_cohorts,
    publication_matches,
    support_labels,
)


def attach_support_evidence(report, evidence):
    attachment = evidence.get("support_policy")
    rows = report.get("observations", [])
    subjects = [review_subject(r) for r in sorted(rows, key=lambda r: r["case_id"])]
    if (
        report.get("backend") != "live"
        or not rows
        or None in subjects
        or evidence.get("support_subjects_sha256") != digest(subjects)
    ):
        return False
    policy = load_support_policy(
        attachment,
        runtime_identity=rows[0].get("runtime_identity"),
        scope_kind=rows[0].get("source_scope", {}).get("kind"),
    )
    if policy is None:
        return False
    accepted = reviewed_policy_record(attachment)
    from .evidence_support_activation import activation_matches

    try:
        return activation_matches(report, evidence, accepted, policy)
    except (KeyError, ValueError, TypeError, AttributeError, OSError):
        return False


def support_gate_errors(report, verified):
    errors = []
    if verified.get("support_policy_verified") is not True:
        errors.append("missing independently accepted support policy")
    else:
        try:
            accepted = reviewed_policy_record(
                report["independent_evidence"]["support_policy"]
            )
            normal = [
                r
                for r in report["observations"]
                if r.get("quality_aggregation") == "normal"
            ]
            if report["mode"] not in accepted["criteria"]["modes"]:
                raise ValueError("unreviewed support arm")
            measure_cohorts(
                normal,
                {**accepted["criteria"], "modes": [report["mode"]]},
                splits=(report["split"],),
            )
        except (KeyError, ValueError, TypeError, OSError):
            errors.append("current support cohort acceptance unproven")
    for row in report.get("observations", []):
        if row.get("quality_aggregation") != "normal":
            continue
        labels = support_labels(row)
        support = row.get("support_decision")
        if not publication_matches(row):
            errors.append("support final publication missing or mismatched")
        if not support_delivery_matches(row):
            errors.append("support final delivery binding missing or mismatched")
        if (
            not isinstance(support, dict)
            or support.get("state")
            not in ("complete", "partial", "insufficient", "unknown")
            or not row.get("primary_question")
            or row.get("primary_question") != row.get("support_question")
            or not row.get("runtime_identity")
            or not row.get("source_scope")
            or row.get("corpus_sha256") != report.get("corpus_sha256")
        ):
            errors.append("support identity or decision missing")
        elif support.get("state") == "complete" and (
            support.get("assessment_basis")
            not in ("deterministic", "validated_semantic")
            or support.get("unresolved_aspects")
            or not support.get("supported_aspects")
        ):
            errors.append("unsupported complete decision")
        if (
            isinstance(support, dict)
            and support.get("assessment_basis") == "validated_semantic"
            and (
                verified.get("support_policy_verified") is not True
                or support.get("calibration_id") != row.get("policy_identity")
            )
        ):
            errors.append("unverified semantic support calibration")
        if labels is None:
            errors.append("missing independent support labels")
        elif labels["false_complete"] or labels["unsupported_answer"]:
            errors.append("independently observed unsupported completion/answer")
        elif not labels["abstained"] and not all(labels[k] for k in DIMENSIONS):
            errors.append("independent support dimension failed")
        if any(row.get(k) for k in ("citation_violations", "authorization_violations")):
            errors.append("support authorization/citation violation")
        lanes = row.get("lane_observations", {})
        if any(
            lanes.get(k, {}).get("status") != "observed"
            for k in ("retrieval", "graph", "source_generation")
        ):
            errors.append(
                "support retrieval/graph/source-generation observations missing"
            )
    return sorted(set(errors))


def support_observation_metrics(row):
    support = row.get("support_decision") or {}
    labels = support_labels(row)
    return {
        "state": support.get("state", "unknown"),
        "assessment_basis": support.get("assessment_basis", "unvalidated"),
        "supported_aspect_count": len(support.get("supported_aspects", [])),
        "unresolved_aspect_count": len(support.get("unresolved_aspects", [])),
        "abstention_outcome": (
            "correct" if labels["abstention_correct"] else "incorrect"
        )
        if labels and labels["abstained"]
        else "unknown",
        "independent_labels": labels,
        "publication_matches": publication_matches(row),
        "answer_sha256": text_digest(row["answer"])
        if isinstance(row.get("answer"), str)
        else None,
        "delivered_sha256": digest(
            {"spans": row.get("delivered"), "sdk_payloads": row.get("sdk_payloads")}
        ),
        "final_stream_sha256": text_digest(row["final_stream"])
        if isinstance(row.get("final_stream"), str)
        else None,
        "persisted_sha256": text_digest(row["persisted_answer"])
        if isinstance(row.get("persisted_answer"), str)
        else None,
        # Relevance/nDCG never substitutes for calibration.
        "calibration_reliability": "unmeasured",
        "operational_coverage": row.get("coverage"),
        "lane_observations": row.get("lane_observations", {}),
    }


def paired_domain_errors(groups):
    from .evidence_quality_comparison import paired_bootstrap_interval
    from .evidence_quality_eval import MODES

    reasons = []
    domains = {g["baseline"].get("domain") for g in groups} - {None}
    for domain in sorted(domains):
        domain_rows = [g for g in groups if g["baseline"].get("domain") == domain]
        # Preserve marginal checks; gains there cannot conceal a crossed decline.
        cohorts = {domain: domain_rows}
        cohorts.update(
            {
                f"{domain}/{depth}": [
                    g for g in domain_rows if g["baseline"].get("depth") == depth
                ]
                for depth in ("routine", "deeper")
            }
        )
        for cohort_name, cohort in cohorts.items():
            for mode in MODES[1:]:
                for key in ("support_recall", "faithfulness", "qualification_accuracy"):
                    pairs = [(g["baseline"].get(key), g[mode].get(key)) for g in cohort]
                    deltas = [
                        b - a for a, b in pairs if a is not None and b is not None
                    ]
                    interval = paired_bootstrap_interval(deltas)
                    if len(deltas) != len(pairs) or not deltas or interval["lower"] < 0:
                        reasons.append(
                            f"{mode}: domain {cohort_name}/{key} decline or unknown"
                        )
    return reasons
