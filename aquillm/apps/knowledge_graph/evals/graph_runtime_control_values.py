"""Exact observed values, using production bounds without filling missing fields."""

import math
import re

from lib.knowledge_graph import retrieval_config as config

from .graph_gate_observations import require

RUNTIME_FIELDS = (
    "graph_algorithm",
    "graph_topology_backend",
    "graph_direct_enabled",
    "graph_extended_enabled",
    "memgraph_traversal_enabled",
    "graph_overall_timeout_ms",
    "graph_direct_timeout_ms",
    "graph_extended_timeout_ms",
    "graph_direct_max_candidates",
    "graph_extended_max_candidates",
    "graph_direct_max_depth",
    "graph_extended_max_depth",
    "graph_direct_max_nodes",
    "graph_extended_max_nodes",
    "graph_direct_max_edges",
    "graph_extended_max_edges",
    "graph_direct_max_seeds",
    "graph_extended_max_seeds",
    "query_extractor_timeout_ms",
    "query_max_bytes",
    "query_max_codepoints",
    "query_max_spans",
    "direct_embedding_enabled",
    "direct_min_similarity",
    "direct_winner_margin",
    "query_extractor_model",
    "query_extractor_model_revision",
    "query_extractor_build_hash",
    "query_extractor_expected_schema_version",
    "query_extractor_expected_schema_checksum",
    "ppr_restart_mode",
    "projection_schema_version",
    "projection_format_version",
    "memgraph_image",
)


def validate_runtime(value):
    reason = "graph_runtime_control_invalid"
    require(type(value) is dict and set(value) == set(RUNTIME_FIELDS), reason)
    for field, observed in value.items():
        key = "KG_" + field.upper()
        if key in config._BOOL_DEFAULTS:
            require(type(observed) is bool, reason)
        elif key in config._INT_RULES:
            _, lower, upper = config._INT_RULES[key]
            require(type(observed) is int and lower <= observed <= upper, reason)
        elif key in config._FLOAT_RULES:
            _, lower, upper = config._FLOAT_RULES[key]
            require(
                type(observed) is float
                and math.isfinite(observed)
                and lower <= observed <= upper,
                reason,
            )
        else:
            require(type(observed) is str, reason)
            require(config._parse_text({key: observed}, key, "") == observed, reason)
    for field, choices in {
        "graph_algorithm": ("ppr_projected_v1",),
        "graph_topology_backend": ("memgraph",),
        "projection_schema_version": ("collection-graph-v1",),
        "projection_format_version": ("projection-v1",),
        "ppr_restart_mode": ("fixed", "shadow", "adaptive"),
    }.items():
        require(value[field] in choices, reason)
    require(
        value["graph_direct_timeout_ms"] <= value["graph_overall_timeout_ms"]
        and value["graph_extended_timeout_ms"] <= value["graph_overall_timeout_ms"]
        and value["direct_winner_margin"] <= value["direct_min_similarity"],
        reason,
    )
    require(
        not value["direct_embedding_enabled"] or value["graph_direct_enabled"], reason
    )
    require(
        not (value["graph_direct_enabled"] or value["graph_extended_enabled"])
        or value["memgraph_traversal_enabled"],
        reason,
    )
    require(
        bool(value["memgraph_image"]) and not re.search(r"\s", value["memgraph_image"]),
        reason,
    )
    for field, pattern in (
        ("query_extractor_model", config._MODEL),
        ("query_extractor_model_revision", config._LOWER_REVISION),
        ("query_extractor_expected_schema_version", config._LOWER_TOKEN),
    ):
        require(pattern.fullmatch(value[field]) is not None, reason)
    for field in (
        "query_extractor_build_hash",
        "query_extractor_expected_schema_checksum",
    ):
        require(
            not value[field]
            or config._LOWER_CHECKSUM.fullmatch(value[field]) is not None,
            reason,
        )
    if value["graph_direct_enabled"]:
        require(bool(value["query_extractor_build_hash"]), reason)
        for field, expected in {
            "query_extractor_model": config.QUERY_EXTRACTOR_MODEL,
            "query_extractor_model_revision": config.QUERY_EXTRACTOR_MODEL_REVISION,
            "query_extractor_expected_schema_version": config.QUERY_SCHEMA_VERSION,
            "query_extractor_expected_schema_checksum": config.QUERY_SCHEMA_CHECKSUM,
        }.items():
            require(value[field] == expected, reason)
