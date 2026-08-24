from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]
COMPOSE_FILES = tuple(ROOT / "deploy" / "compose" / name for name in (
    "base.yml", "development.yml", "no_gpu_dev.yml", "production.yml",
))
DEFAULTS = {
    "KG_SCHEMA_GENERATION_ENABLED": "${KG_SCHEMA_GENERATION_ENABLED:-0}",
    "KG_SCHEMA_GENERATION_MAX_CHUNKS": "${KG_SCHEMA_GENERATION_MAX_CHUNKS:-32}",
    "KG_SCHEMA_GENERATION_MAX_CHARACTERS": "${KG_SCHEMA_GENERATION_MAX_CHARACTERS:-48000}",
    "KG_SCHEMA_GENERATION_TIMEOUT_SECONDS": "${KG_SCHEMA_GENERATION_TIMEOUT_SECONDS:-180}",
}


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda path: path.name)
def test_schema_generation_settings_reach_the_isolated_graph_worker(path: Path) -> None:
    environment = yaml.safe_load(path.read_text(encoding="utf-8"))["services"]["worker_knowledge_graph"]["environment"]
    for name, expected in DEFAULTS.items():
        if name == "KG_SCHEMA_GENERATION_ENABLED" and path.name in {"development.yml", "no_gpu_dev.yml"}:
            expected = "1"
        assert environment[name] == expected
