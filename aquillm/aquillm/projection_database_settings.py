"""Strict, default-off database aliases for the graph projection runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalConfigError,
    _postgres_identity,
)

_ENGINE = "django_prometheus.db.backends.postgresql"


def _disabled_database() -> dict[str, object]:
    return {
        "ENGINE": _ENGINE,
        "NAME": "projection_alias_disabled",
        "USER": "",
        "PASSWORD": "",
        "HOST": "127.0.0.1",
        "PORT": "1",
        "OPTIONS": {"connect_timeout": 2},
    }


def _projection_database(
    name: str,
    expected_role: str,
    source: Mapping[str, str],
) -> dict[str, object]:
    raw = source.get(name, "")
    if not raw:
        return _disabled_database()
    try:
        identity = _postgres_identity(name, raw)
        parsed = urlsplit(raw)
        valid = identity[0] == expected_role
    except (HybridRetrievalConfigError, ValueError):
        valid = False
    if not valid:
        if source.get("KG_MEMGRAPH_PROJECTION_ENABLED") == "1":
            raise ValueError(f"{name} must be a canonical PostgreSQL DSN")
        return _disabled_database()
    return {
        "ENGINE": _ENGINE,
        "NAME": identity[3],
        "USER": identity[0],
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": identity[1],
        "PORT": str(identity[2]),
        "OPTIONS": {"connect_timeout": 5},
    }


def projection_databases(
    source: Mapping[str, str] = os.environ,
) -> dict[str, dict[str, object]]:
    return {
        "projection_source": _projection_database(
            "KG_PROJECTION_POSTGRES_SOURCE_DSN",
            "aquillm_projection_source",
            source,
        ),
        "projection_state": _projection_database(
            "KG_PROJECTION_POSTGRES_STATE_DSN",
            "aquillm_projection_state",
            source,
        ),
    }
