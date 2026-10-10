"""Separate public synthetic plumbing scenarios; no quality adoption labels."""

import json
from pathlib import Path


def load_cases():
    return json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))


def execution_inputs(case):
    return dict(case["inputs"])
