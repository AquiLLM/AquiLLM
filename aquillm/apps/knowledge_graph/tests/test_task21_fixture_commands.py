from __future__ import annotations

import importlib
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import CommandError, call_command

ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
SOURCE_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = SOURCE_ROOT.parent
_ML_IMPORT_PROBE = """
import importlib
import sys

forbidden_prefixes = (
    "gliner2",
    "lib.knowledge_graph.extractors",
    "torch",
    "transformers",
)
before = set(sys.modules)
importlib.import_module(sys.argv[1])
imported = set(sys.modules) - before
forbidden = sorted(
    name
    for name in imported
    if any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in forbidden_prefixes
    )
)
if forbidden:
    print("forbidden=" + ",".join(forbidden[:8]))
    raise SystemExit(17)
print("ok")
"""


def _ontology_checksum() -> str:
    from apps.knowledge_graph.services.ontology import load_ontology

    return load_ontology(ONTOLOGY_PATH).checksum


def _run_ml_import_probe(
    module: str, *python_paths: Path
) -> subprocess.CompletedProcess:
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "WINDIR")
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _ML_IMPORT_PROBE, module],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.django_db(transaction=True)
def test_activate_ontology_dry_run_outputs_identity_without_writing() -> None:
    from apps.knowledge_graph.models import OntologyVersion

    output = StringIO()
    call_command(
        "activate_knowledge_graph_ontology",
        "--path",
        "research-v1.yaml",
        "--expected-checksum",
        _ontology_checksum(),
        "--dry-run",
        stdout=output,
    )

    assert output.getvalue() == (f"version=1.0.0 checksum={_ontology_checksum()}\n")
    assert not OntologyVersion.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_activate_ontology_rejects_checksum_mismatch_without_writing() -> None:
    from apps.knowledge_graph.models import OntologyVersion

    with pytest.raises(CommandError, match="checksum"):
        call_command(
            "activate_knowledge_graph_ontology",
            "--path",
            "research-v1.yaml",
            "--expected-checksum",
            "0" * 64,
            "--dry-run",
        )

    assert not OntologyVersion.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_activate_ontology_requires_confirmation_before_writing() -> None:
    from apps.knowledge_graph.models import OntologyVersion

    with pytest.raises(CommandError, match="--yes"):
        call_command(
            "activate_knowledge_graph_ontology",
            "--path",
            "research-v1.yaml",
            "--expected-checksum",
            _ontology_checksum(),
        )

    assert not OntologyVersion.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_activate_ontology_rejects_path_escape_without_writing() -> None:
    from apps.knowledge_graph.models import OntologyVersion

    with pytest.raises(CommandError, match="ontology directory"):
        call_command(
            "activate_knowledge_graph_ontology",
            "--path",
            "../evals/extraction_cases.yaml",
            "--expected-checksum",
            _ontology_checksum(),
            "--dry-run",
        )

    assert not OntologyVersion.objects.exists()


def test_activate_ontology_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    command_module = importlib.import_module(
        "apps.knowledge_graph.management.commands.activate_knowledge_graph_ontology"
    )
    ontology_root = tmp_path / "ontologies"
    ontology_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text(ONTOLOGY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    link = ontology_root / "linked.yaml"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    monkeypatch.setattr(command_module, "_ONTOLOGY_ROOT", ontology_root)

    with pytest.raises(CommandError, match="ontology directory"):
        call_command(
            "activate_knowledge_graph_ontology",
            "--path",
            "linked.yaml",
            "--expected-checksum",
            _ontology_checksum(),
            "--dry-run",
        )


@pytest.mark.django_db(transaction=True)
def test_activate_ontology_is_idempotent_with_one_active_row() -> None:
    from apps.knowledge_graph.models import OntologyVersion

    checksum = _ontology_checksum()
    for _attempt in range(2):
        output = StringIO()
        call_command(
            "activate_knowledge_graph_ontology",
            "--path",
            "research-v1.yaml",
            "--expected-checksum",
            checksum,
            "--yes",
            stdout=output,
        )
        assert output.getvalue() == f"version=1.0.0 checksum={checksum}\n"

    assert OntologyVersion.objects.count() == 1
    assert OntologyVersion.objects.get().status == OntologyVersion.Status.ACTIVE


def test_activate_ontology_command_imports_no_ml_runtime() -> None:
    result = _run_ml_import_probe(
        "apps.knowledge_graph.management.commands.activate_knowledge_graph_ontology",
        SOURCE_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "ok\n"
    assert result.stderr == ""


def test_ml_import_probe_detects_a_controlled_forbidden_import(tmp_path) -> None:
    (tmp_path / "gliner2.py").write_text("", encoding="utf-8")
    (tmp_path / "task21_probe_target.py").write_text(
        "import gliner2\n", encoding="utf-8"
    )

    result = _run_ml_import_probe("task21_probe_target", tmp_path)

    assert result.returncode == 17
    assert result.stdout == "forbidden=gliner2\n"
    assert result.stderr == ""
