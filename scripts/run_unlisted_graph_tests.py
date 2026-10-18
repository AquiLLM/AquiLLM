"""Run graph regressions not already selected by a dedicated CI step."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = "aquillm/apps/knowledge_graph/tests/"


def remaining_tests(root=ROOT):
    workflow = (root / ".github/workflows/test-backend-frontend.yml").read_text()
    selected = set(re.findall(re.escape(GRAPH) + r"test_[A-Za-z0-9_]+\.py", workflow))
    available = {
        p.relative_to(root).as_posix() for p in (root / GRAPH).glob("test_*.py")
    }
    return sorted(available - selected)


if __name__ == "__main__":
    remaining = remaining_tests()
    print(f"Graph files without a dedicated CI step: {len(remaining)}", flush=True)
    if remaining:
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *remaining,
                    "--ds=aquillm.settings_projection_test",
                    "-q",
                    "--tb=short",
                ],
                cwd=ROOT,
            )
        )
