"""Child startup must select the pytest database before app readiness hooks."""
import json
import subprocess
import sys

import pytest

from .worker_handoff_support import WorkerHarness


@pytest.mark.django_db(transaction=True)
def test_child_selects_settings_and_connection_database_before_django_setup(tmp_path):
    harness = WorkerHarness(tmp_path)
    observations = tmp_path / "startup.json"
    probe = r'''
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import django
from django.apps import apps
from django.conf import settings
from django.db import connections
from tests.integration import worker_handoff_bootstrap

original_setup = django.setup
observed = []

def setup_observer(*args, **kwargs):
    selected = settings.DATABASES["default"]["NAME"]
    connected = connections["default"].settings_dict["NAME"]
    assert selected == "test_rag_test", f"settings DB at setup entry: {selected}"
    assert connected == "test_rag_test", f"connection DB at setup entry: {connected}"
    assert os.environ["POSTGRES_NAME"] == "rag_test"
    observed.append({"settings": selected, "connection": connected, "ready": apps.ready})
    return original_setup(*args, **kwargs)

sys.argv = ["worker_handoff_bootstrap", "discover"]
with patch.object(django, "setup", side_effect=setup_observer):
    worker_handoff_bootstrap.main()
Path(os.environ["HANDOFF_STARTUP"]).write_text(json.dumps(observed))
'''
    result = subprocess.run([sys.executable, "-c", probe],
        env={**harness.environment, "HANDOFF_STARTUP": str(observations)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(observations.read_text())[0] == {
        "settings": "test_rag_test", "connection": "test_rag_test", "ready": False,
    }
    receipt = json.loads(harness.receipt.read_text())
    assert receipt["database"] == "test_rag_test" and receipt["eager"] is False
    assert "documents.build_embedding_generation" in receipt["discovered"]
