"""CI covers newly added graph files and rejects missing required services."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_new_graph_file_is_selected_without_workflow_edit(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "unlisted_graph_tests",
        ROOT / "scripts/run_unlisted_graph_tests.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workflow = tmp_path / ".github/workflows/test-backend-frontend.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "run: pytest aquillm/apps/knowledge_graph/tests/test_existing.py"
    )
    tests = tmp_path / module.GRAPH
    tests.mkdir(parents=True)
    for name in ("test_existing.py", "test_new.py"):
        (tests / name).touch()
    assert module.remaining_tests(tmp_path) == [module.GRAPH + "test_new.py"]


@pytest.mark.parametrize("body,code", [
    ("pass", 0), ("pytest.skip('unavailable')", 1),
    ("pytest.xfail('unavailable')", 1),
])
def test_required_service_skip_changes_job_result(tmp_path, body, code):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_service.py").write_text(
        "import pytest\ndef test_service():\n    " + body + "\n",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.pytest_required_services",
            "test_service.py",
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )
    assert result.returncode == code, result.stdout.decode(errors="replace")
