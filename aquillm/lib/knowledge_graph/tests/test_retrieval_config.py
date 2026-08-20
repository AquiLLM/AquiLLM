# ruff: noqa: E501
"""Fail-closed contracts for hybrid graph retrieval configuration."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from lib.knowledge_graph import retrieval_config as config

# fmt: off
DEFAULTS: dict[str, object] = {
    "memgraph_projection_enabled": False,
    "memgraph_traversal_enabled": False,
    "graph_direct_enabled": False,
    "graph_extended_enabled": False,
    "graph_topology_backend": "memgraph",
    "graph_algorithm": "ppr_projected_v1",
    "memgraph_image": "memgraph/memgraph-mage:3.8.1",
    "memgraph_uri": "",
    "memgraph_database": "memgraph",
    "memgraph_query_username": "",
    "memgraph_query_password": "",
    "memgraph_projection_username": "",
    "memgraph_projection_password": "",
    "projection_postgres_source_dsn": "",
    "projection_postgres_state_dsn": "",
    "projection_queue": "knowledge_graph_projection",
    "projection_schema_version": "collection-graph-v1",
    "projection_format_version": "projection-v1",
    "projection_identifier_hmac_key": "",
    "projection_identifier_key_version": "",
    "projection_batch_size": 500,
    "projection_lease_seconds": 300,
    "projection_max_attempts": 5,
    "projection_retention": 2,
    "projection_max_lag_seconds": 300,
    "query_extractor_url": "",
    "query_extractor_bearer_token": "",
    "query_extractor_model": "fastino/gliner2-base-v1",
    "query_extractor_model_revision": "8437ba583a733d87f56ae902f3b197934eedd58e",
    "query_extractor_expected_schema_version": "query-entities-v1",
    "query_extractor_expected_schema_checksum": "",
    "query_extractor_timeout_ms": 75,
    "query_max_bytes": 4096,
    "query_max_codepoints": 2048,
    "query_max_spans": 32,
    "graph_overall_timeout_ms": 300,
    "graph_direct_timeout_ms": 125,
    "graph_extended_timeout_ms": 125,
    "graph_direct_max_seeds": 32,
    "graph_direct_max_depth": 2,
    "graph_direct_max_nodes": 200,
    "graph_direct_max_edges": 1000,
    "graph_direct_max_candidates": 20,
    "graph_extended_max_seeds": 64,
    "graph_extended_max_depth": 2,
    "graph_extended_max_nodes": 200,
    "graph_extended_max_edges": 1000,
    "graph_extended_max_candidates": 20,
    "graph_fusion_rrf_k": 60,
    "direct_embedding_enabled": False,
    "direct_min_similarity": 0.80,
    "direct_winner_margin": 0.05,
    "graph_eval_parity_backend": "postgres",
}

INT_LIMITS = {
    "KG_PROJECTION_BATCH_SIZE": (1, 5000),
    "KG_PROJECTION_LEASE_SECONDS": (10, 3600),
    "KG_PROJECTION_MAX_ATTEMPTS": (1, 20),
    "KG_PROJECTION_RETENTION": (1, 50),
    "KG_PROJECTION_MAX_LAG_SECONDS": (1, 86400),
    "KG_QUERY_EXTRACTOR_TIMEOUT_MS": (10, 1000),
    "KG_QUERY_MAX_BYTES": (1, 16384),
    "KG_QUERY_MAX_CODEPOINTS": (1, 8192),
    "KG_QUERY_MAX_SPANS": (1, 128),
    "KG_GRAPH_OVERALL_TIMEOUT_MS": (25, 5000),
    "KG_GRAPH_DIRECT_TIMEOUT_MS": (10, 5000),
    "KG_GRAPH_EXTENDED_TIMEOUT_MS": (10, 5000),
    "KG_GRAPH_DIRECT_MAX_SEEDS": (1, 64),
    "KG_GRAPH_DIRECT_MAX_DEPTH": (1, 2),
    "KG_GRAPH_DIRECT_MAX_NODES": (1, 200),
    "KG_GRAPH_DIRECT_MAX_EDGES": (1, 1000),
    "KG_GRAPH_DIRECT_MAX_CANDIDATES": (1, 20),
    "KG_GRAPH_EXTENDED_MAX_SEEDS": (1, 64),
    "KG_GRAPH_EXTENDED_MAX_DEPTH": (1, 2),
    "KG_GRAPH_EXTENDED_MAX_NODES": (1, 200),
    "KG_GRAPH_EXTENDED_MAX_EDGES": (1, 1000),
    "KG_GRAPH_EXTENDED_MAX_CANDIDATES": (1, 20),
}


def _projection() -> dict[str, str]:
    return {
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_MEMGRAPH_URI": "bolt://graph:7687",
        "KG_MEMGRAPH_PROJECTION_USERNAME": "writer",
        "KG_MEMGRAPH_PROJECTION_PASSWORD": "projection-secret",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": "postgresql://reader@pg/source",
        "KG_PROJECTION_POSTGRES_STATE_DSN": "postgresql://cas@pg/state",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "hmac-secret",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "task21-key-v1",
    }


def _traversal() -> dict[str, str]:
    return {
        "KG_MEMGRAPH_TRAVERSAL_ENABLED": "1",
        "KG_MEMGRAPH_URI": "bolt://graph:7687",
        "KG_MEMGRAPH_QUERY_USERNAME": "reader",
        "KG_MEMGRAPH_QUERY_PASSWORD": "query-secret",
    }


def _direct() -> dict[str, str]:
    return {
        **_traversal(),
        "KG_GRAPH_DIRECT_ENABLED": "1",
        "KG_QUERY_EXTRACTOR_URL": "https://extractor.internal/v1/entities",
        "KG_QUERY_EXTRACTOR_BEARER_TOKEN": "extractor-secret",
        "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": config.QUERY_SCHEMA_CHECKSUM,
    }


def test_defaults_and_fields_are_exact_and_off() -> None:
    settings = config.load_hybrid_retrieval_settings({})
    assert {field.name: getattr(settings, field.name) for field in dataclasses.fields(settings)} == DEFAULTS
    assert settings.graph_topology_backend == "memgraph"

@pytest.mark.parametrize("key", ("KG_MEMGRAPH_PROJECTION_ENABLED", "KG_MEMGRAPH_TRAVERSAL_ENABLED", "KG_GRAPH_DIRECT_ENABLED", "KG_GRAPH_EXTENDED_ENABLED", "KG_DIRECT_EMBEDDING_ENABLED"))
@pytest.mark.parametrize("bad", ("true", "false", "yes", "01", " 1", "1 ", "", True, 1, None))
def test_booleans_accept_only_exact_string_bits(key: str, bad: object) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match=key):
        config.load_hybrid_retrieval_settings(cast(dict[str, str], {key: bad}))

@pytest.mark.parametrize(("key", "limits"), INT_LIMITS.items())
def test_integer_limits_accept_both_edges(key: str, limits: tuple[int, int]) -> None:
    low, high = limits
    overrides = {key: str(low)}
    if key == "KG_GRAPH_OVERALL_TIMEOUT_MS":
        overrides |= {"KG_GRAPH_DIRECT_TIMEOUT_MS": "10", "KG_GRAPH_EXTENDED_TIMEOUT_MS": "10"}
    if key.endswith("_TIMEOUT_MS") and key != "KG_GRAPH_OVERALL_TIMEOUT_MS":
        overrides["KG_GRAPH_OVERALL_TIMEOUT_MS"] = "5000"
    config.load_hybrid_retrieval_settings(overrides)
    config.load_hybrid_retrieval_settings({**overrides, key: str(high)})

@pytest.mark.parametrize(("key", "limits"), INT_LIMITS.items())
def test_integer_limits_reject_both_off_by_one_values(key: str, limits: tuple[int, int]) -> None:
    for bad in (limits[0] - 1, limits[1] + 1):
        with pytest.raises(config.HybridRetrievalConfigError, match=key):
            config.load_hybrid_retrieval_settings({key: str(bad)})

@pytest.mark.parametrize("bad", ("01", "+1", " 1", "1 ", "1.0", "1e1", "", True, 1))
def test_integer_spellings_and_types_are_canonical(bad: object) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match="KG_QUERY_MAX_SPANS"):
        config.load_hybrid_retrieval_settings(cast(dict[str, str], {"KG_QUERY_MAX_SPANS": bad}))

@pytest.mark.parametrize("key", ("KG_DIRECT_MIN_SIMILARITY", "KG_DIRECT_WINNER_MARGIN"))
@pytest.mark.parametrize("good", ("0", "0.0", "0.80", "1", "1.0"))
def test_ratios_accept_strict_bounded_decimal_spellings(key: str, good: str) -> None:
    overrides = {key: good}
    if key.endswith("WINNER_MARGIN"):
        overrides["KG_DIRECT_MIN_SIMILARITY"] = "1"
    elif float(good) < 0.05:
        overrides["KG_DIRECT_WINNER_MARGIN"] = "0"
    config.load_hybrid_retrieval_settings(overrides)

@pytest.mark.parametrize("bad", ("-0.1", "1.1", ".8", "00.8", "+0.8", " 0.8", "0.8 ", "1e-1", "nan", "inf", True, 1.0))
def test_ratios_reject_noncanonical_or_nonstring_values(bad: object) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match="KG_DIRECT_MIN_SIMILARITY"):
        config.load_hybrid_retrieval_settings(cast(dict[str, str], {"KG_DIRECT_MIN_SIMILARITY": bad}))

@pytest.mark.parametrize("key", ("KG_MEMGRAPH_URI", "KG_MEMGRAPH_DATABASE", "KG_MEMGRAPH_PROJECTION_USERNAME", "KG_MEMGRAPH_PROJECTION_PASSWORD", "KG_PROJECTION_POSTGRES_SOURCE_DSN", "KG_PROJECTION_POSTGRES_STATE_DSN", "KG_PROJECTION_IDENTIFIER_HMAC_KEY", "KG_PROJECTION_IDENTIFIER_KEY_VERSION"))
def test_projection_requires_each_connection_and_secret_independently(key: str) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match=key):
        config.load_hybrid_retrieval_settings({**_projection(), key: ""})

@pytest.mark.parametrize("key", ("KG_MEMGRAPH_URI", "KG_MEMGRAPH_DATABASE", "KG_MEMGRAPH_QUERY_USERNAME", "KG_MEMGRAPH_QUERY_PASSWORD"))
def test_traversal_requires_each_connection_and_secret_independently(key: str) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match=key):
        config.load_hybrid_retrieval_settings({**_traversal(), key: ""})

@pytest.mark.parametrize(("key", "bad"), (("KG_MEMGRAPH_TRAVERSAL_ENABLED", "0"), ("KG_QUERY_EXTRACTOR_URL", ""), ("KG_QUERY_EXTRACTOR_BEARER_TOKEN", ""), ("KG_QUERY_EXTRACTOR_MODEL", "other/model"), ("KG_QUERY_EXTRACTOR_MODEL_REVISION", "a" * 40), ("KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION", "other-v1"), ("KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM", "b" * 64)))
def test_direct_requires_exact_extractor_contract(key: str, bad: str) -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match=key):
        config.load_hybrid_retrieval_settings({**_direct(), key: bad})

def test_extended_and_embedding_dependencies_fail_closed() -> None:
    with pytest.raises(config.HybridRetrievalConfigError, match="KG_MEMGRAPH_TRAVERSAL_ENABLED"):
        config.load_hybrid_retrieval_settings({"KG_GRAPH_EXTENDED_ENABLED": "1"})
    with pytest.raises(config.HybridRetrievalConfigError, match="KG_GRAPH_DIRECT_ENABLED"):
        config.load_hybrid_retrieval_settings({"KG_DIRECT_EMBEDDING_ENABLED": "1"})

def test_disabled_paths_allow_empty_connections_and_both_branches_enable_independently() -> None:
    assert config.load_hybrid_retrieval_settings({"KG_MEMGRAPH_DATABASE": ""}).memgraph_database == ""
    assert config.load_hybrid_retrieval_settings(_direct()).graph_direct_enabled
    assert config.load_hybrid_retrieval_settings({**_traversal(), "KG_GRAPH_EXTENDED_ENABLED": "1"}).graph_extended_enabled

@pytest.mark.parametrize("overrides", ({"KG_GRAPH_DIRECT_TIMEOUT_MS": "301"}, {"KG_GRAPH_EXTENDED_TIMEOUT_MS": "301"}, {"KG_DIRECT_MIN_SIMILARITY": "0.4", "KG_DIRECT_WINNER_MARGIN": "0.5"}, {"KG_GRAPH_FUSION_RRF_K": "59"}, {"KG_GRAPH_FUSION_RRF_K": "61"}, {**_projection(), "KG_PROJECTION_POSTGRES_SOURCE_DSN": "postgresql://same/db", "KG_PROJECTION_POSTGRES_STATE_DSN": "postgresql://same/db"}))
def test_cross_field_constraints(overrides: dict[str, str]) -> None:
    with pytest.raises(config.HybridRetrievalConfigError):
        config.load_hybrid_retrieval_settings(overrides)

@pytest.mark.parametrize("overrides", ({"KG_GRAPH_TOPOLOGY_BACKEND": "postgres"}, {"KG_GRAPH_ALGORITHM": "ppr_v1"}, {"KG_GRAPH_EVAL_PARITY_BACKEND": "memgraph"}, {"KG_QUERY_EXTRACTOR_MODEL_REVISION": "A" * 40}, {"KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": "A" * 64}, {"KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION": "Bad Schema"}, {"KG_PROJECTION_IDENTIFIER_KEY_VERSION": "Bad Key"}, {"KG_PROJECTION_POSTGRES_SOURCE_DSN": "not-a-dsn"}, {"KG_PROJECTION_QUEUE": " bad"}, {"KG_PROJECTION_QUEUE": "bad\x7f"}, {"KG_PROJECTION_QUEUE": "bad\ud800"}, {"KG_MEMGRAPH_IMAGE": "bad image"}, {"KG_MEMGRAPH_QUERY_USERNAME": "bad user"}, {"KG_MEMGRAPH_URI": "http://graph"}, {"KG_MEMGRAPH_URI": "bolt://user:secret@graph"}, {"KG_QUERY_EXTRACTOR_URL": "file:///tmp/model"}))
def test_tokens_urls_and_revisions_are_strict(overrides: dict[str, str]) -> None:
    with pytest.raises(config.HybridRetrievalConfigError):
        config.load_hybrid_retrieval_settings(overrides)

def test_source_shape_and_unknown_graph_keys_fail_closed() -> None:
    for source in ({"KG_GRAPH_DIREC_ENABLED": "1"}, {1: "x"}, {"UNRELATED": 1}):
        with pytest.raises(config.HybridRetrievalConfigError):
            config.load_hybrid_retrieval_settings(cast(Any, source))
    original = {"UNRELATED": "kept"}
    assert config.load_hybrid_retrieval_settings(original)
    assert original == {"UNRELATED": "kept"}

def test_secrets_are_redacted_from_repr_and_errors_but_affect_equality() -> None:
    canary = "DO-NOT-LOG-CANARY"
    valid = {**_projection(), "KG_MEMGRAPH_PROJECTION_PASSWORD": canary}
    settings = config.load_hybrid_retrieval_settings(valid)
    assert canary not in repr(settings)
    changed = config.load_hybrid_retrieval_settings({**valid, "KG_MEMGRAPH_PROJECTION_PASSWORD": "different"})
    assert settings != changed
    with pytest.raises(config.HybridRetrievalConfigError) as caught:
        config.load_hybrid_retrieval_settings({**valid, "KG_PROJECTION_POSTGRES_STATE_DSN": valid["KG_PROJECTION_POSTGRES_SOURCE_DSN"]})
    assert canary not in str(caught.value)
    assert all(value not in repr(settings) for key, value in valid.items() if "PASSWORD" in key or "DSN" in key or "HMAC" in key)
    direct = config.load_hybrid_retrieval_settings({**_direct(), "KG_MEMGRAPH_QUERY_PASSWORD": canary, "KG_QUERY_EXTRACTOR_BEARER_TOKEN": canary})
    assert canary not in repr(direct)

def test_evaluation_backend_requires_exact_private_capability() -> None:
    settings = config.load_hybrid_retrieval_settings({})
    assert config.select_evaluation_topology_backend(settings, capability=config._EVALUATION_BACKEND_CAPABILITY) == "postgres"
    for capability in (None, object(), type("Lookalike", (), {})()):
        with pytest.raises(config.HybridRetrievalConfigError, match="capability"):
            config.select_evaluation_topology_backend(settings, capability=capability)
    assert settings.graph_topology_backend == "memgraph"

def test_settings_are_frozen_slotted_and_import_isolated() -> None:
    settings = config.load_hybrid_retrieval_settings({})
    assert not hasattr(settings, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.graph_algorithm = "changed"  # type: ignore[misc]
    script = "import sys; import lib.knowledge_graph.retrieval_config; print(int('django' in sys.modules or 'neo4j' in sys.modules))"
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0"
# fmt: on
