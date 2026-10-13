"""Synthetic gate positives test arithmetic/binding, never human acceptance."""

import json
from copy import deepcopy
from hashlib import sha256

IDENTITY = (
    "stable_identifiers",
    "local_definitions",
    "alias_collisions",
    "version_distinctions",
)
STRATA = (
    "homonym",
    "relation_intent",
    "conflicting_evidence",
    "hub",
    "cross_generation_bridge",
    "partial_readiness",
)


def digest(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fixture_report():
    from apps.knowledge_graph.evals.graph_effective_controls import (
        graph_effective_controls,
    )
    from apps.knowledge_graph.tests.graph_review_fixtures import (
        expand_modes,
        rebind,
        rehash,
    )

    baseline = graph_effective_controls()
    baseline["remediation_mode"] = "legacy"
    rehash(baseline)
    candidate = deepcopy(baseline)
    candidate["remediation_mode"] = "v2"
    rehash(candidate)
    binding = {
        key: digest(key)
        for key in (
            "runtime",
            "model_runtime",
            "corpus",
            "budget",
            "graph_budget",
            "source_generation",
        )
    }
    binding.update(revision="a" * 40, algorithm_signature=digest("graph-v2-test"))
    decisions = [
        dict(
            decision_id=f"d{i}",
            pair_ids=[f"entity-{i}-a", f"entity-{i}-b"],
            strata=[IDENTITY[i % 4]],
            automatic=True,
            candidate=True,
            expected_same=True,
            known_conflict=False,
        )
        for i in range(600)
    ]
    decisions.append(
        dict(
            decision_id="conflict",
            pair_ids=["conflict-a", "conflict-b"],
            strata=[IDENTITY[2]],
            automatic=False,
            candidate=True,
            expected_same=False,
            known_conflict=True,
        )
    )
    queries = []
    for i in range(150):
        tags = [STRATA[i % 3], STRATA[3 + i % 3]]
        labels = dict(relevant=["good"], no_answer=False, coverage="partial")
        inputs = dict(
            query_id=f"q{i}", question=f"Question {i}", scope_sha256=digest(i)
        )
        arms = {}
        for arm in ("baseline", "candidate"):
            arms[arm] = dict(
                binding=deepcopy(binding),
                inputs=deepcopy(inputs),
                ranked=["bad", "good"] if arm == "baseline" else ["good"],
                repeated_ranked=["bad", "good"] if arm == "baseline" else ["good"],
                abstained=False,
                coverage="partial",
                exhaustive=False,
                authorization_violations=0,
                citation_violations=0,
                completion="completed",
                cache_state="cold",
                provider_cache="unknown",
                latency_ms=10,
                latency_budget_ms=20,
                request_algorithm_signature=digest(f"request-{i}-{arm}"),
                frontier_budget=dict(
                    policy_signature=digest("synthetic-budget-policy"),
                    max_queries=64,
                    max_rows=5000,
                    max_bytes=1048576,
                    max_identities=200,
                    max_edges=500,
                    max_depth=2,
                    queries_used=2,
                    rows_used=5,
                    bytes_used=500,
                    identities_used=2,
                    edges_used=1,
                    accounting_complete=True,
                    uncertain=False,
                ),
            )
        for arm, observation in arms.items():
            observation["frontier_invocations"] = [
                dict(
                    invocation_id=f"q{i}-{arm}",
                    query_sha256=digest(inputs["question"]),
                    requested_subset_signature=digest(i),
                    frontier_budget=deepcopy(observation["frontier_budget"]),
                    request_algorithm_signature=observation[
                        "request_algorithm_signature"
                    ],
                )
            ]
        queries.append(dict(query_id=f"q{i}", strata=tags, labels=labels, **arms))
    # Positive recall improvement too: baseline misses one half of each stratum.
    for i, row in enumerate(queries):
        if i % 2 == 0:
            row["baseline"]["ranked"] = row["baseline"]["repeated_ranked"] = ["bad"]
        if i >= 144:
            row["labels"]["relevant"] = []
            row["labels"]["no_answer"] = True
            for arm in ("baseline", "candidate"):
                row[arm]["ranked"] = row[arm]["repeated_ranked"] = []
                row[arm]["abstained"] = True
    report = seal(
        dict(
            schema="graph-evaluation-v2",
            backend="live",
            split="heldout",
            binding=binding,
            baseline_binding=deepcopy(binding),
            intended_improvement="relation_intent",
            preregistered=True,
            heldout_independent=True,
            predeclared_at=1,
            heldout_opened_at=2,
            development_decision_ids=[],
            development_query_ids=[],
            policy_sha256=digest(
                dict(
                    min_automatic_precision=0.995,
                    min_automatic_precision_lower_95=0.99,
                    min_automatic_pairs=600,
                    min_query_pairs=150,
                    min_pairs_per_stratum=50,
                    max_metric_decline=0.0,
                )
            ),
            decisions=decisions,
            queries=queries,
            controls=dict(baseline=baseline, candidate=candidate),
            treatment={"remediation_mode": {"baseline": "legacy", "candidate": "v2"}},
        )
    )
    rebind(report)
    return expand_modes(report)


def seal(report):
    report["labels_sha256"] = digest(
        {
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
                for d in report["decisions"]
            ],
            "queries": [
                {k: q[k] for k in ("query_id", "strata", "labels")}
                for q in report["queries"]
            ],
        }
    )
    return report
