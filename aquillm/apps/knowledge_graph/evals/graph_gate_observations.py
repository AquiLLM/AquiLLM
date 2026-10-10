"""Strict graph study originals; labels never enter execution inputs."""

import hashlib
import json
import math
import re

IDENTITY_STRATA = (
    "stable_identifiers",
    "local_definitions",
    "alias_collisions",
    "version_distinctions",
)
RETRIEVAL_STRATA = (
    "homonym",
    "relation_intent",
    "conflicting_evidence",
    "hub",
    "cross_generation_bridge",
    "partial_readiness",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=dict
        ).encode()
    ).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def ids(value):
    require(
        type(value) is list and all(type(v) is str and v for v in value), "invalid_ids"
    )
    require(len(set(value)) == len(value), "duplicate_ids")
    return value


def tags(value, allowed):
    require(bool(ids(value)) and set(value) <= set(allowed), "invalid_strata")


def number(value):
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "invalid_number",
    )
    return value


def binding(value):
    require(
        type(value) is dict
        and set(value)
        == {
            "revision",
            "algorithm_signature",
            "runtime",
            "model_runtime",
            "corpus",
            "budget",
            "graph_budget",
            "source_generation",
        },
        "invalid_comparison_binding",
    )
    require(
        type(value["revision"]) is str
        and re.fullmatch("[0-9a-f]{40}", value["revision"]),
        "invalid_revision",
    )
    require(
        all(
            sha(value[k])
            for k in (
                "runtime",
                "model_runtime",
                "corpus",
                "budget",
                "graph_budget",
                "source_generation",
            )
        ),
        "unknown_comparison_binding",
    )
    require(
        sha(value["algorithm_signature"]),
        "unknown_algorithm_signature",
    )


def validate_report(report):
    from .graph_mode_observations import validate_modes
    from .graph_study_controls import validate_study_controls

    require(
        set(report)
        == {
            "schema",
            "backend",
            "split",
            "binding",
            "baseline_binding",
            "controls",
            "treatment",
            "intended_improvement",
            "preregistered",
            "heldout_independent",
            "predeclared_at",
            "heldout_opened_at",
            "development_decision_ids",
            "development_query_ids",
            "policy_sha256",
            "labels_sha256",
            "decisions",
            "queries",
        },
        "graph_report_fields_unknown",
    )

    binding(report["binding"])
    binding(report["baseline_binding"])
    validate_study_controls(report)
    require(
        all(
            report["binding"][k] == report["baseline_binding"][k]
            for k in (
                "revision",
                "model_runtime",
                "corpus",
                "budget",
                "graph_budget",
                "source_generation",
            )
        ),
        "unmatched_model_corpus_budget_source",
    )
    require(
        report["split"] == "heldout" and report["heldout_independent"] is True,
        "heldout_independence_unknown",
    )
    require(
        report["preregistered"] is True
        and number(report["predeclared_at"]) < number(report["heldout_opened_at"]),
        "preregistration_unknown",
    )
    require(
        report["intended_improvement"] in RETRIEVAL_STRATA,
        "improvement_stratum_unknown",
    )
    decisions, queries = report["decisions"], report["queries"]
    require(type(decisions) is list and type(queries) is list, "observations_unknown")
    seen, pairs = set(), set()
    for row in decisions:
        require(
            type(row) is dict
            and set(row)
            == {
                "decision_id",
                "pair_ids",
                "strata",
                "automatic",
                "candidate",
                "expected_same",
                "known_conflict",
            },
            "invalid_identity_observation",
        )
        identity = row["decision_id"]
        require(
            type(identity) is str and identity and identity not in seen,
            "duplicate_or_invalid_decision",
        )
        seen.add(identity)
        pair = tuple(sorted(ids(row["pair_ids"])))
        require(len(pair) == 2 and pair not in pairs, "duplicate_or_invalid_pair")
        pairs.add(pair)
        tags(row["strata"], IDENTITY_STRATA)
        require(
            all(
                type(row[k]) is bool
                for k in ("automatic", "candidate", "expected_same", "known_conflict")
            ),
            "identity_labels_unknown",
        )
        require(
            not (row["known_conflict"] and row["expected_same"]),
            "contradictory_identity_labels",
        )
    require(
        not seen.intersection(ids(report["development_decision_ids"])),
        "heldout_decision_leak",
    )
    seen, inputs_seen, invocations_seen = set(), set(), set()
    for row in queries:
        require(
            type(row) is dict
            and set(row) == {"query_id", "strata", "labels", "executions"},
            "invalid_query_observation",
        )
        identity = row["query_id"]
        require(
            type(identity) is str and identity and identity not in seen,
            "duplicate_or_invalid_query",
        )
        seen.add(identity)
        tags(row["strata"], RETRIEVAL_STRATA)
        labels = row["labels"]
        require(
            type(labels) is dict
            and set(labels) == {"relevant", "no_answer", "coverage"},
            "query_labels_unknown",
        )
        ids(labels["relevant"])
        require(
            type(labels["no_answer"]) is bool
            and labels["no_answer"] == (not labels["relevant"]),
            "contradictory_no_answer_labels",
        )
        require(
            labels["coverage"] in ("complete", "partial", "unavailable"),
            "coverage_label_unknown",
        )
        inputs = validate_modes(row, report, invocations_seen)
        query_key = (inputs["question"], inputs["scope_sha256"])
        require(query_key not in inputs_seen, "duplicate_query_content")
        inputs_seen.add(query_key)
    require(
        not seen.intersection(ids(report["development_query_ids"])),
        "heldout_query_leak",
    )
    labels = {
        "decisions": [
            {
                k: d[k]
                for k in (
                    "decision_id",
                    "pair_ids",
                    "strata",
                    "expected_same",
                    "known_conflict",
                )
            }
            for d in decisions
        ],
        "queries": [
            {k: q[k] for k in ("query_id", "strata", "labels")} for q in queries
        ],
    }
    require(report["labels_sha256"] == digest(labels), "labels_digest_mismatch")
