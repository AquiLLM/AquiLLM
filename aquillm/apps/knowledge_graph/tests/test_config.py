"""Shipping configuration contracts for the knowledge-graph overlay."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from apps.knowledge_graph.retrieval import ppr as ppr_module
from apps.knowledge_graph.retrieval.ppr import (
    PPRAlgorithmConfig,
    graph_algorithm_signature,
)
from lib.knowledge_graph import retrieval_config
from lib.knowledge_graph.config import (
    DEFAULT_ARTIFACT_KEEP_SUPERSEDED,
    DEFAULT_ARTIFACT_RETENTION_DAYS,
    DEFAULT_EXTRACTOR_PROVIDER,
    DEFAULT_GLINER2_BATCH_SIZE,
    DEFAULT_GLINER2_CACHE_DIR,
    DEFAULT_GLINER2_DEVICE,
    DEFAULT_GLINER2_MODEL,
    DEFAULT_GLINER2_REVISION,
    get_eval_bypass_allowed,
    load_extraction_settings,
    load_retention_settings,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY_SETTING_DEFAULTS: dict[str, object] = {
    "KG_EXTRACTION_QUEUE": "knowledge-graph-extraction",
    "KG_OVERLAY_ENABLED": False,
    "KG_OVERLAY_ALGORITHM": "ppr_v1",
    "KG_OVERLAY_RRF_K": 60,
    "KG_OVERLAY_MAX_SEEDS": 64,
    "KG_OVERLAY_MAX_SCOPE_DOCUMENTS": 10_000,
    "KG_OVERLAY_MAX_SCOPE_COLLECTIONS": 128,
    "KG_OVERLAY_MAX_HOPS": 2,
    "KG_OVERLAY_MAX_FANOUT": 10,
    "KG_OVERLAY_MAX_NODES": 200,
    "KG_OVERLAY_MAX_EDGES": 1_000,
    "KG_OVERLAY_MAX_EVIDENCE_ROWS": 3_000,
    "KG_OVERLAY_MAX_EVIDENCE_PER_EDGE": 3,
    "KG_OVERLAY_MAX_MENTIONS_PER_ENTITY": 2,
    "KG_OVERLAY_PPR_RESTART": 0.20,
    "KG_OVERLAY_PPR_ITERATIONS": 8,
    "KG_OVERLAY_MAX_CANDIDATES": 20,
    "KG_OVERLAY_MAX_PER_DOCUMENT": 3,
    "KG_OVERLAY_TIMEOUT_MS": 150,
}


def _run_settings_script(
    script: str, overrides: dict[str, str], names: tuple[str, ...]
) -> str:
    environment = os.environ.copy()
    for name in names:
        environment.pop(name, None)
    environment.update(overrides)
    environment |= {
        "DJANGO_DEBUG": "1",
        "DJANGO_SETTINGS_MODULE": "aquillm.settings",
        "PYTHONPATH": str(_PROJECT_ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _load_django_graph_settings(
    overrides: dict[str, str],
    names: tuple[str, ...] = tuple(_OVERLAY_SETTING_DEFAULTS),
) -> dict[str, object]:
    script = (
        "import json\n"
        "from aquillm import settings\n"
        f"names = {names!r}\n"
        "print(json.dumps({name: getattr(settings, name) for name in names}, "
        "sort_keys=True, default=repr))\n"
    )
    return json.loads(_run_settings_script(script, overrides, names))


def _django_startup_accepts_but_retrieval_rejects(
    overrides: dict[str, str],
) -> bool:
    script = (
        "import django, json\n"
        "django.setup()\n"
        "from apps.knowledge_graph.retrieval import get_graph_expansion_config\n"
        "try:\n"
        "    get_graph_expansion_config()\n"
        "except (TypeError, ValueError):\n"
        "    valid = False\n"
        "else:\n"
        "    valid = True\n"
        "print(json.dumps({'valid': valid}))\n"
    )
    output = _run_settings_script(script, overrides, tuple(_OVERLAY_SETTING_DEFAULTS))
    return bool(json.loads(output.splitlines()[-1])["valid"])


def test_extraction_retention_and_eval_controls_are_safe_by_default() -> None:
    extraction = load_extraction_settings({})
    retention = load_retention_settings({})

    assert extraction.build_enabled is False
    assert extraction.provider == DEFAULT_EXTRACTOR_PROVIDER == "gliner2_local"
    assert extraction.fail_open is True
    assert extraction.model_id == DEFAULT_GLINER2_MODEL == "fastino/gliner2-base-v1"
    assert extraction.model_revision == DEFAULT_GLINER2_REVISION
    assert extraction.device == DEFAULT_GLINER2_DEVICE == "cpu"
    assert extraction.batch_size == DEFAULT_GLINER2_BATCH_SIZE == 8
    assert extraction.cache_dir == DEFAULT_GLINER2_CACHE_DIR
    assert extraction.local_files_only is False
    assert retention.retention_days == DEFAULT_ARTIFACT_RETENTION_DAYS == 30
    assert retention.keep_superseded == DEFAULT_ARTIFACT_KEEP_SUPERSEDED == 2
    assert get_eval_bypass_allowed({}) is False


def test_django_settings_expose_off_by_default_overlay_and_queue_contract() -> None:
    assert _load_django_graph_settings({}) == _OVERLAY_SETTING_DEFAULTS


def test_django_exposes_bounded_hybrid_defaults_despite_hostile_ambient() -> None:
    expected = retrieval_config.load_django_hybrid_retrieval_settings({})
    encoded = json.loads(json.dumps(expected, default=repr))
    hostile = {"KG_BUILD_ENABLED": "1", "KG_GRAPH_DIRECT_ENABLEDD": "1"}

    assert _load_django_graph_settings({}, tuple(expected)) == encoded
    assert _load_django_graph_settings(hostile, tuple(expected)) == encoded
    assert not any(value for name, value in encoded.items() if name.endswith("ENABLED"))


def test_django_settings_parse_exact_overlay_ceiling_values() -> None:
    overrides = {
        name: "1" if value is True else str(value)
        for name, value in _OVERLAY_SETTING_DEFAULTS.items()
    }
    overrides["KG_OVERLAY_ENABLED"] = "1"

    observed = _load_django_graph_settings(overrides)

    assert observed == {**_OVERLAY_SETTING_DEFAULTS, "KG_OVERLAY_ENABLED": True}


def test_django_settings_extraction_queue_is_configurable() -> None:
    observed = _load_django_graph_settings(
        {"KG_EXTRACTION_QUEUE": "isolated-knowledge-graph"}
    )

    assert observed["KG_EXTRACTION_QUEUE"] == "isolated-knowledge-graph"


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("KG_OVERLAY_ALGORITHM", "global_pagerank"),
        ("KG_OVERLAY_MAX_NODES", "not-an-integer"),
    ),
)
def test_invalid_overlay_environment_preserves_startup_but_rejects_graph_work(
    name: str,
    value: str,
) -> None:
    assert not _django_startup_accepts_but_retrieval_rejects(
        {"KG_OVERLAY_ENABLED": "1", name: value}
    )


_HARD_LIMITS = {
    "rrf_k": 1_000,
    "max_seeds": 64,
    "max_scope_documents": 10_000,
    "max_scope_collections": 128,
    "max_hops": 2,
    "max_fanout": 10,
    "max_nodes": 200,
    "max_edges": 1_000,
    "max_evidence_rows": 3_000,
    "max_evidence_per_edge": 3,
    "max_mentions_per_entity": 2,
    "ppr_iterations": 8,
    "max_candidates": 20,
    "max_per_document": 3,
    "timeout_ms": 150,
}


def test_ppr_config_accepts_every_shipping_ceiling_together() -> None:
    config = PPRAlgorithmConfig(
        canonical_resolver_version="canonical-resolution-v1",
        **_HARD_LIMITS,
    )

    assert {name: getattr(config, name) for name in _HARD_LIMITS} == _HARD_LIMITS


@pytest.mark.parametrize(("field_name", "maximum"), tuple(_HARD_LIMITS.items()))
def test_ppr_config_rejects_each_ceiling_plus_one(
    field_name: str,
    maximum: int,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        PPRAlgorithmConfig(
            canonical_resolver_version="canonical-resolution-v1",
            **{field_name: maximum + 1},
        )


@pytest.mark.parametrize(
    "restart",
    (True, "0.2", float("nan"), float("inf"), float("-inf"), 0.0, 1.0),
)
def test_ppr_restart_is_an_exact_finite_open_interval(restart: object) -> None:
    with pytest.raises(ValueError, match="ppr_restart"):
        PPRAlgorithmConfig(
            canonical_resolver_version="canonical-resolution-v1",
            ppr_restart=restart,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"max_scope_documents": 127, "max_scope_collections": 128},
        {"max_nodes": 9, "max_fanout": 10, "max_edges": 91},
        {"max_evidence_rows": 2, "max_evidence_per_edge": 3},
        {"max_candidates": 2, "max_per_document": 3},
    ),
)
def test_ppr_config_rejects_every_cross_field_violation(
    overrides: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        PPRAlgorithmConfig(
            canonical_resolver_version="canonical-resolution-v1",
            **overrides,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("canonical_resolver_version", "canonical-resolution-v2"),
        ("rrf_k", 59),
        ("max_seeds", 63),
        ("max_scope_documents", 9_999),
        ("max_scope_collections", 127),
        ("max_hops", 1),
        ("max_fanout", 9),
        ("max_nodes", 199),
        ("max_edges", 999),
        ("max_evidence_rows", 2_999),
        ("max_evidence_per_edge", 2),
        ("max_mentions_per_entity", 1),
        ("ppr_restart", 0.21),
        ("ppr_iterations", 7),
        ("max_candidates", 19),
        ("max_per_document", 2),
        ("timeout_ms", 149),
    ),
)
def test_algorithm_signature_changes_with_every_effective_setting(
    field_name: str,
    replacement: object,
) -> None:
    baseline = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")
    changed = replace(baseline, **{field_name: replacement})

    assert graph_algorithm_signature(changed) != graph_algorithm_signature(baseline)


@pytest.mark.parametrize(
    "constant_name",
    ("ALGORITHM_VERSION", "TRANSITION_VERSION", "EVIDENCE_VERSION", "SEED_VERSION"),
)
def test_algorithm_signature_changes_with_every_frozen_version(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
) -> None:
    config = PPRAlgorithmConfig(canonical_resolver_version="canonical-resolution-v1")
    baseline = graph_algorithm_signature(config)

    monkeypatch.setattr(ppr_module, constant_name, f"changed-{constant_name.lower()}")

    assert graph_algorithm_signature(config) != baseline
