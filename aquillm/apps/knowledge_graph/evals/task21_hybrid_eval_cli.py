"""Write one validated Task21 five-arm observation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .task21_hybrid_eval import build_task21_hybrid_report, canonical_json_bytes


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--freshness", type=Path, required=True)
    parser.add_argument("--backend-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))
    cases = (
        raw_cases["cases"]
        if isinstance(raw_cases, Mapping) and "cases" in raw_cases
        else raw_cases
    )
    report = build_task21_hybrid_report(
        cases=cases,
        observations=_json(args.observations),
        freshness=_json(args.freshness),
        backend_parity=_json(args.backend_parity),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        handle.write(canonical_json_bytes(report) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    temporary.rename(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"task21_hybrid_eval={args.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
