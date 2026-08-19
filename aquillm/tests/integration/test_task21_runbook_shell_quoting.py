from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNBOOK = (
    REPO / "docs" / "documents" / "operations" / "knowledge-graph-overlay-runbook.md"
)


def _procedure() -> str:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    fenced = runbook.split("Run this Bash procedure", 1)[1]
    fenced = fenced.split("```bash", 1)[1].split("```", 1)[0]
    return "\n".join(
        line[3:] if line.startswith("   ") else line for line in fenced.splitlines()
    )


def _inspection_validator() -> str:
    section = _procedure().split(
        'inspection="$(kg_eval_python manage.py inspect_knowledge_graph', 1
    )[1]
    return section.split("kg_eval_python -c '\n", 1)[1].split(
        '\n\' "$inspection" "$request_id"', 1
    )[0]


def _run_inspection_validator(report: dict[str, object], request_id: str):
    return subprocess.run(
        [sys.executable, "-c", _inspection_validator(), json.dumps(report), request_id],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pid_attestation_survives_shell_quoting() -> None:
    procedure = _procedure()
    marker = '"$container" python3 -c \''
    payload = procedure.split(marker, 1)[1].split("'\ndone", 1)[0]
    rendered = subprocess.run(
        ["bash", "-c", f"printf '%s' '{payload}'"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    compile(rendered, "<runbook-pid-attestation>", "exec")


def test_mandatory_procedure_is_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n"],
        input=_procedure().encode("utf-8"),
        check=True,
        capture_output=True,
    )


def test_reranker_probe_initializes_django_before_model_import() -> None:
    probe = _procedure().split("kg_eval_no_cache_python -c '", 2)[2]
    probe = probe.split("strict_local_reranker=ok", 1)[0]

    settings = probe.index('os.environ.setdefault("DJANGO_SETTINGS_MODULE"')
    setup = probe.index("django.setup()")
    model_import = probe.index("from apps.documents.models import TextChunk")
    assert settings < setup < model_import


def test_runtime_artifact_paths_are_scoped_to_the_unique_run() -> None:
    procedure = _procedure()

    assert (
        'KG_EVAL_MANIFEST="/app/artifacts/kg-eval-fixture-manifest-'
        '$KG_EVAL_RUN_ID.json"' in procedure
    )
    assert (
        'KG_EVAL_REPORT="/app/artifacts/kg-eval-comparison-$KG_EVAL_RUN_ID.json"'
        in procedure
    )


def test_eval_inspection_accepts_only_private_success_lifecycle() -> None:
    request_id = "00000000-0000-4000-8000-000000000001"
    artifact = {
        "pk": 11,
        "evaluation_only": True,
        "rebuild_request_id": request_id,
        "status": "superseded",
    }
    build = {
        "pk": 21,
        "artifact_id": 11,
        "evaluation_only": True,
        "rebuild_request_id": request_id,
        "stage": "superseded",
        "status": "cancelled",
        "error_code": "",
        "evaluation_completed": True,
    }
    report = {
        "request_id": request_id,
        "effective_request_id": request_id,
        "status": "succeeded",
        "request_error_code": "",
        "failure_count": 0,
        "truncated": False,
        "artifact_count": 1,
        "build_count": 1,
        "artifact_ids": [11],
        "build_ids": [21],
        "artifacts": [artifact],
        "builds": [build],
    }

    assert _run_inspection_validator(report, request_id).returncode == 0

    bad_artifact = {**report, "artifacts": [{**artifact, "status": "active"}]}
    assert _run_inspection_validator(bad_artifact, request_id).returncode != 0
    duplicate_artifact_row = {**report, "artifacts": [artifact, artifact]}
    assert _run_inspection_validator(duplicate_artifact_row, request_id).returncode != 0
    duplicate_build_row = {**report, "builds": [build, build]}
    assert _run_inspection_validator(duplicate_build_row, request_id).returncode != 0
    for mutation in (
        {"build_count": 2},
        {"build_ids": [21, 21]},
    ):
        assert (
            _run_inspection_validator({**report, **mutation}, request_id).returncode
            != 0
        )
    second_artifact = {**artifact, "pk": 12}
    duplicate_artifact_link = {
        **report,
        "artifact_count": 2,
        "build_count": 2,
        "artifact_ids": [11, 12],
        "build_ids": [21, 22],
        "artifacts": [artifact, second_artifact],
        "builds": [build, {**build, "pk": 22}],
    }
    assert (
        _run_inspection_validator(duplicate_artifact_link, request_id).returncode != 0
    )
    for mutation in (
        {"stage": "active"},
        {"status": "succeeded"},
        {"error_code": "private_failure"},
        {"evaluation_completed": False},
        {"artifact_id": 12},
        {"rebuild_request_id": "00000000-0000-4000-8000-000000000002"},
    ):
        bad_build = {**report, "builds": [{**build, **mutation}]}
        assert _run_inspection_validator(bad_build, request_id).returncode != 0
