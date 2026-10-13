"""Offline graph gate attachment CLI. Exit 2 means blocked, never activation."""

import argparse
import json
from pathlib import Path

from .graph_activation_attachment_v2 import graph_activation_attachment_v2


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_graph_report(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("graph report exceeds bounded input")
    report = json.loads(
        raw,
        object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    if type(report) is not dict:
        raise ValueError("graph report must be an object")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = load_graph_report(args.report)
    result = graph_activation_attachment_v2(report)
    from .run_kg_eval import atomic_write_json

    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "eligible": result["result"]["eligible"],
                "blocking_reasons": result["result"]["blocking_reasons"],
            }
        )
    )
    return 0 if result["result"]["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
