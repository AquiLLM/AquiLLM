"""Strict, default-off database aliases for the graph projection runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import unquote, urlsplit

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


def _projection_database(name: str, source: Mapping[str, str]) -> dict[str, object]:
    raw = source.get(name, "")
    if not raw:
        return _disabled_database()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
        database_name = parsed.path.removeprefix("/")
        valid = (
            parsed.scheme in {"postgres", "postgresql"}
            and parsed.hostname is not None
            and parsed.username is not None
            and parsed.password is not None
            and bool(database_name)
            and port is not None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        if source.get("KG_MEMGRAPH_PROJECTION_ENABLED") == "1":
            raise ValueError(f"{name} must be a canonical PostgreSQL DSN")
        return _disabled_database()
    return {
        "ENGINE": _ENGINE,
        "NAME": unquote(database_name),
        "USER": unquote(parsed.username),
        "PASSWORD": unquote(parsed.password),
        "HOST": parsed.hostname,
        "PORT": str(port),
        "OPTIONS": {"connect_timeout": 5},
    }


def projection_databases(
    source: Mapping[str, str] = os.environ,
) -> dict[str, dict[str, object]]:
    return {
        "projection_source": _projection_database(
            "KG_PROJECTION_POSTGRES_SOURCE_DSN", source
        ),
        "projection_state": _projection_database(
            "KG_PROJECTION_POSTGRES_STATE_DSN", source
        ),
    }
