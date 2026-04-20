"""Contracts for RAG cache settings and Django CACHES defaults."""
from __future__ import annotations

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


def test_caches_default_locmem_without_redis_url():
    assert settings.CACHES["default"]["BACKEND"].endswith("LocMemCache")


@pytest.mark.parametrize(
    "env_redis",
    ["redis://127.0.0.1:6379/2"],
)
def test_caches_redis_when_url_set_subprocess(env_redis):
    """Fresh interpreter picks Redis backend when DJANGO_CACHE_REDIS_URL is set."""
    project = Path(__file__).resolve().parents[2]
    code = (
        "import os, django\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aquillm.settings')\n"
        f"os.environ['DJANGO_CACHE_REDIS_URL'] = {env_redis!r}\n"
        "os.environ.setdefault('SECRET_KEY', 'test-subprocess-secret-key-xxxxxxxx')\n"
        "django.setup()\n"
        "from django.conf import settings as s\n"
        "assert 'RedisCache' in s.CACHES['default']['BACKEND']\n"
    )
    env = {
        **os.environ,
        "DJANGO_CACHE_REDIS_URL": env_redis,
        "SECRET_KEY": "test-subprocess-secret-key-xxxxxxxx",
        "OPENAI_API_KEY": "test",
        "GEMINI_API_KEY": "test",
        "GOOGLE_OAUTH2_CLIENT_ID": "test-oauth-client-id",
        "GOOGLE_OAUTH2_CLIENT_SECRET": "test-oauth-client-secret",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
