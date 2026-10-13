"""Contracts for RAG cache settings and Django CACHES defaults."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.conf import settings


def test_rag_cache_flags_default_off():
    assert settings.RAG_CACHE_ENABLED is False
    assert settings.RAG_QUERY_EMBED_TTL_SECONDS == 300
    assert settings.RAG_DOC_ACCESS_TTL_SECONDS == 60
    assert settings.RAG_IMAGE_DATA_URL_TTL_SECONDS == 120
    assert settings.RAG_RERANK_RESULT_TTL_SECONDS == 45
    assert settings.RAG_RERANK_CAPABILITY_TTL_SECONDS == 900


def test_token_efficiency_defaults():
    assert settings.TOKEN_EFFICIENCY_ENABLED is False
    assert settings.PROMPT_BUDGET_CONTEXT_LIMIT == 0


def test_context_packer_defaults():
    assert settings.CONTEXT_PACKER_ENABLED is False
    assert settings.CONTEXT_BUDGET_TOOL_EVIDENCE_TOKENS == 1400


def _cache_configuration(env_redis):
    """Read base configuration in a fresh child without starting Django/apps/IO."""
    project = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OTEL_ENABLED="0")
    if env_redis is None:
        env.pop("DJANGO_CACHE_REDIS_URL", None)
    else:
        env["DJANGO_CACHE_REDIS_URL"] = env_redis
    code = (
        "import json\n"
        "from django.apps import apps\n"
        "assert not apps.ready\n"
        "from aquillm import settings as base\n"
        "assert not apps.ready\n"
        "print(json.dumps(base.CACHES['default']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=project, env=env,
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_caches_default_locmem_without_redis_url():
    config = _cache_configuration(None)
    assert config == {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "aquillm-default",
    }


def test_caches_redis_when_url_set_subprocess():
    # Use only the owned test service; configuration inspection opens no client.
    location = "redis://redis:6379/1"
    config = _cache_configuration(location)
    assert config["BACKEND"] == "django.core.cache.backends.redis.RedisCache"
    assert config["LOCATION"] == location
    assert config["TIMEOUT"] == 300


@pytest.mark.parametrize(
    "configured, expected",
    [(None, ""), ("", ""), ("a" * 64, "a" * 64), (" \t" + "b" * 64 + "\n", "b" * 64)],
)
def test_document_embedding_space_environment_reaches_django_settings(
    configured, expected,
):
    """An opt-in process key must reach consumers without a settings override."""
    env = dict(os.environ, OTEL_ENABLED="0", DJANGO_SETTINGS_MODULE="aquillm.settings")
    if configured is None:
        env.pop("DOCUMENT_EMBEDDING_SPACE_KEY", None)
    else:
        env["DOCUMENT_EMBEDDING_SPACE_KEY"] = configured
    code = (
        "import json\n"
        "from django.apps import apps\n"
        "from django.conf import settings\n"
        "print(json.dumps(getattr(settings, 'DOCUMENT_EMBEDDING_SPACE_KEY', None)))\n"
        "assert not apps.ready\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == expected
