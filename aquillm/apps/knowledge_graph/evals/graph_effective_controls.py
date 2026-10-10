"""Capture actual secret-free graph settings for the existing review subject."""

from dataclasses import asdict

from .graph_gate_observations import digest, require, sha


def graph_effective_controls():
    from django.conf import settings

    from apps.knowledge_graph.resolution import resolver_version_for_policy
    from apps.knowledge_graph.retrieval.expansion import _load_algorithm_config
    from lib.knowledge_graph.config import load_identity_policy

    policy = load_identity_policy()
    algorithm = _load_algorithm_config()
    mode = settings.KG_GRAPH_REMEDIATION_MODE
    # Explicit allowlist avoids endpoints, credentials and connection settings.
    from .graph_runtime_control_values import RUNTIME_FIELDS

    result = dict(
        identity_policy=policy,
        remediation_mode=mode,
        resolver_versions={
            tier: resolver_version_for_policy(tier, policy)
            for tier in ("document", "collection", "canonical")
        },
        algorithm_config=asdict(algorithm),
        algorithm_signature=effective_signature(algorithm, mode),
        runtime_observations={
            key: getattr(settings, "KG_" + key.upper()) for key in RUNTIME_FIELDS
        },
    )
    result["runtime_sha256"] = digest(result)
    return result


def effective_signature(config, mode):
    from apps.knowledge_graph.retrieval.ppr import graph_algorithm_signature
    from apps.knowledge_graph.retrieval.projected_relation_ranking_v2 import (
        relation_algorithm_signature_v2,
    )

    require(mode in ("legacy", "shadow", "v2"), "graph_applicability_unknown")
    return (
        graph_algorithm_signature(config)
        if mode == "legacy"
        else relation_algorithm_signature_v2(config)
    )


def graph_budget_digest(controls):
    """Caps cannot be reclassified as the intentional graph-policy treatment."""
    algorithm = controls["algorithm_config"]
    runtime = controls["runtime_observations"]
    return digest(
        {
            "algorithm": {
                k: v
                for k, v in algorithm.items()
                if k.startswith("max_") or k in ("timeout_ms", "ppr_iterations")
            },
            "runtime": {
                k: v
                for k, v in runtime.items()
                if "_max_" in k or k.endswith("_timeout_ms")
            },
        }
    )


def validate_controls(value, *, observed_only=False):
    from apps.knowledge_graph.resolution import resolver_version_for_policy
    from apps.knowledge_graph.retrieval.ppr import (
        PPRAlgorithmConfig,
    )

    require(
        type(value) is dict
        and set(value)
        == {
            "identity_policy",
            "remediation_mode",
            "resolver_versions",
            "algorithm_config",
            "algorithm_signature",
            "runtime_observations",
            "runtime_sha256",
        },
        "graph_applicability_unknown",
    )
    policy = value["identity_policy"]
    require(
        policy in ("legacy", "contextual_v2")
        and value["remediation_mode"] in ("legacy", "shadow", "v2"),
        "graph_applicability_unknown",
    )
    require(
        value["resolver_versions"]
        == {
            tier: resolver_version_for_policy(tier, policy)
            for tier in ("document", "collection", "canonical")
        },
        "graph_applicability_unknown",
    )
    algorithm = PPRAlgorithmConfig(**value["algorithm_config"])
    require(
        set(value["algorithm_config"]) == set(asdict(algorithm)),
        "graph_algorithm_config_incomplete",
    )
    require(
        algorithm.max_hops == 2
        and algorithm.ppr_iterations == 8
        and value["algorithm_signature"]
        == effective_signature(algorithm, value["remediation_mode"]),
        "graph_applicability_unknown",
    )
    require(
        sha(value["runtime_sha256"])
        and value["runtime_sha256"]
        == digest({k: v for k, v in value.items() if k != "runtime_sha256"}),
        "graph_applicability_unknown",
    )
    from .graph_runtime_control_values import validate_runtime

    validate_runtime(value["runtime_observations"])
    require(
        algorithm.canonical_resolver_version
        in {
            resolver_version_for_policy("canonical", "legacy"),
            resolver_version_for_policy("canonical", policy),
        },
        "graph_resolver_config_unknown",
    )
    return value if observed_only else graph_effective_controls()
