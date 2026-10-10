"""Frozen public adversarial supplement: regression expectations, not human data."""

import json
from pathlib import Path

from .evidence_quality_eval import digest

SPLIT_DIGESTS = {
    "development": "e30a7bd9b5fadb4637c1f5d80603698ccf490930f296c92e31d7d98b00a69056",
    "heldout": "fd3f98e3ad1f803dbf1d1eb95c0547cca01d614df1d9589a236f6968dcb87b16",
}


def load_support_cases(path=None):
    root = json.loads(
        Path(
            path or Path(__file__).with_name("evidence_support_remediation_cases.json")
        ).read_text(encoding="utf-8")
    )
    cases = root["cases"]
    if (
        root.get("independent_human_labels") is not False
        or root.get("activation_eligible") is not False
    ):
        raise ValueError("synthetic support fixtures cannot supply independent labels")
    if len({c["case_id"] for c in cases}) != len(cases) or len(cases) != 32:
        raise ValueError("support supplement cases changed")
    for split, frozen in SPLIT_DIGESTS.items():
        if digest([c for c in cases if c["split"] == split]) != frozen:
            raise ValueError("frozen support supplement changed")
    return cases
