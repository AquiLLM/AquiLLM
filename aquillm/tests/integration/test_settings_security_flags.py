"""Regression tests for safe defaults (debug toolbar, Celery serialization)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from django.conf import settings


def test_celery_accept_content_excludes_pickle():
    assert "pickle" not in settings.CELERY_ACCEPT_CONTENT
    assert settings.CELERY_ACCEPT_CONTENT == ["json"]


def test_debug_toolbar_not_installed_when_debug_off_subprocess():
    """Fresh interpreter with DJANGO_DEBUG=0 must not load debug_toolbar."""
    project = Path(__file__).resolve().parents[2]
    code = (
        "import os, django\n"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aquillm.settings')\n"
        "os.environ['DJANGO_DEBUG'] = '0'\n"
        "os.environ.setdefault('SECRET_KEY', 'test-subprocess-secret-key-xxxxxxxx')\n"
        "django.setup()\n"
        "from django.conf import settings as s\n"
        "assert 'debug_toolbar' not in s.INSTALLED_APPS\n"
    )
    env = {
        **os.environ,
        "DJANGO_DEBUG": "0",
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
