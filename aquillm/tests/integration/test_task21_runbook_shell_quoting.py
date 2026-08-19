from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNBOOK = (
    REPO / "docs" / "documents" / "operations" / "knowledge-graph-overlay-runbook.md"
)


def test_pid_attestation_survives_shell_quoting() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    fenced = runbook.split("Run this Bash procedure", 1)[1]
    fenced = fenced.split("```bash", 1)[1].split("```", 1)[0]
    procedure = "\n".join(
        line[3:] if line.startswith("   ") else line for line in fenced.splitlines()
    )
    marker = '"$container" python3 -c \''
    payload = procedure.split(marker, 1)[1].split("'\ndone", 1)[0]
    rendered = subprocess.run(
        ["bash", "-c", f"printf '%s' '{payload}'"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    compile(rendered, "<runbook-pid-attestation>", "exec")
