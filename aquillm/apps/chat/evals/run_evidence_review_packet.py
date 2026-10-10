"""Export private offline review packets or import completed human responses."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.chat.evals.evidence_operational import expand, load_workload
from apps.chat.evals.evidence_quality_eval import load_cases
from apps.chat.evals.evidence_review_packet import import_reviews, prepare_packet


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


def _write(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _reports(paths):
    return {f"report-{index:03d}": _read(path) for index, path in enumerate(paths, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--observations", type=Path, nargs="+", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Local shuffle seed; not a security token",
    )
    load = commands.add_parser("import")
    load.add_argument("--packet", type=Path, required=True)
    load.add_argument("--bindings", type=Path, required=True)
    load.add_argument("--responses", type=Path, required=True)
    load.add_argument("--observations", type=Path, nargs="+", required=True)
    load.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reports = _reports(args.observations)
        if len(reports) != len(args.observations):
            raise ValueError("duplicate report ID")
        if args.command == "export":
            cases = {
                case["case_id"]: case
                for case in load_cases(
                    Path(__file__).with_name("evidence_quality_cases.json")
                )
            }
            workload = load_workload()
            cases.update(
                {w["workload_id"]: expand(w, workload) for w in workload["workloads"]}
            )
            packet, bindings = prepare_packet(reports, cases, random_seed=args.seed)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _write(args.output_dir / "packet.json", packet)
            _write(args.output_dir / "bindings.json", bindings)
            _write(args.output_dir / "responses.json", packet["response_template"])
            _write(
                args.output_dir / "audit-supplement.json", packet["audit_supplement"]
            )
            (args.output_dir / "review-sheet.md").write_text(
                packet["sheet"], encoding="utf-8"
            )
        else:
            reviews = import_reviews(
                _read(args.packet), _read(args.bindings), reports, _read(args.responses)
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for report_id, report_reviews in reviews.items():
                _write(args.output_dir / f"{report_id}-reviews.json", report_reviews)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"review packet: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
