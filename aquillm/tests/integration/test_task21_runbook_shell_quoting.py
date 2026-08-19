from __future__ import annotations

import subprocess
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


def test_reranker_probe_initializes_django_before_model_import() -> None:
    probe = _procedure().split("kg_eval_no_cache_python -c '", 2)[2]
    probe = probe.split("strict_local_reranker=ok", 1)[0]

    settings = probe.index('os.environ.setdefault("DJANGO_SETTINGS_MODULE"')
    setup = probe.index("django.setup()")
    model_import = probe.index("from apps.documents.models import TextChunk")
    assert settings < setup < model_import
