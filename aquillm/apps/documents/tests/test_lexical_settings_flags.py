"""Environment opt-ins must reach the real lexical ranking controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


FLAGS = (
    "RAG_FULL_TEXT_ENABLED",
    "RAG_RRF_ENABLED",
    "RAG_PROTECTED_EXACT_ENABLED",
)


def _lexical_configuration(values):
    env = dict(os.environ, OTEL_ENABLED="0", DJANGO_SETTINGS_MODULE="aquillm.settings")
    for flag, value in zip(FLAGS, values):
        env.pop(flag, None)
        if value is not None:
            env[flag] = value
    code = (
        "import json\n"
        "from django.apps import apps\n"
        "assert not apps.ready\n"
        "from django.conf import settings\n"
        "from apps.documents.services.chunk_search_rank_policy import controls\n"
        f"flags = {FLAGS!r}\n"
        "result = {'settings': [getattr(settings, flag, None) for flag in flags], "
        "'controls': controls()}\n"
        "assert not apps.ready\n"
        "print(json.dumps(result))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[3], env=env,
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "configured, expected_settings, expected_controls",
    [
        ((None, None, None), (False, False, False), (False, False, False)),
        (("1", None, None), (True, False, False), (True, False, False)),
        ((None, "true", None), (False, True, False), (False, True, False)),
        ((None, None, "yes"), (False, False, True), (False, False, False)),
        ((None, "1", "1"), (False, True, True), (False, True, True)),
        (("1", "1", "1"), (True, True, True), (True, True, True)),
        ((" False ", " 0 ", " oFf\t"), (False, False, False), (False, False, False)),
        ((" YES ", "\tOn ", " TrUe "), (True, True, True), (True, True, True)),
    ],
    ids=(
        "default_off", "full_text_only", "rrf_only", "protected_needs_rrf",
        "rrf_and_protected_without_full_text", "all_enabled",
        "explicit_false_with_whitespace", "true_with_whitespace",
    ),
)
def test_lexical_environment_reaches_settings_and_ranking_controls(
    configured, expected_settings, expected_controls,
):
    config = _lexical_configuration(configured)
    assert config["settings"] == list(expected_settings)
    assert config["controls"] == dict(
        zip(("full_text", "rrf", "protected_exact"), expected_controls)
    )
