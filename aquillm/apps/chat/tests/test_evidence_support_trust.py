"""Fixed administrator manifest rejects ambiguous/self-selected trust."""

import json

import pytest

from apps.chat.evals import evidence_support_trust as trust


def empty_manifest():
    return {
        "schema": "support-review-trust-v1",
        "version": 1,
        "calibration": {},
        "activation": {},
    }


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "bool", "nan", "duplicate", "oversize", "bad_pin", "ambiguous"],
)
def test_manifest_rejects_ambiguous_or_malformed_enrollment(mutation):
    manifest = empty_manifest()
    if mutation == "unknown":
        manifest["candidate_trust"] = True
    elif mutation == "bool":
        manifest["version"] = True
    elif mutation == "nan":
        manifest["version"] = float("nan")
    elif mutation == "bad_pin":
        manifest["calibration"]["a" * 64] = {"reviewer": "self"}
    elif mutation == "ambiguous":
        pin = {
            "reviewer": "synthetic",
            "provenance_sha256": "a" * 64,
            "criteria_sha256": "b" * 64,
            "subjects_sha256": "c" * 64,
        }
        manifest["calibration"]["a" * 64] = pin
        manifest["activation"]["a" * 64] = {
            **pin,
            "revision": "d" * 40,
            "policy_sha256": "e" * 64,
        }
    raw = json.dumps(manifest).encode()
    if mutation == "duplicate":
        raw = raw[:-1] + b', "version": 2}'
    elif mutation == "oversize":
        raw += b" " * trust.MAX_TRUST_BYTES
    with pytest.raises((ValueError, TypeError)):
        trust.parse_manifest(raw)


def test_no_file_is_empty_and_candidate_manifest_path_cannot_enroll(
    tmp_path, monkeypatch
):
    from apps.chat.evals.evidence_support_calibration import load_support_policy

    def missing(path, limit):
        assert path == trust.TRUST_PATH
        raise FileNotFoundError

    monkeypatch.setattr(trust, "_read_bytes", missing)
    assert trust.trusted_manifest()["calibration"] == {}
    candidate = tmp_path / "self.json"
    candidate.write_text(json.dumps(empty_manifest()))
    assert (
        load_support_policy(
            {
                "artifact": str(candidate),
                "artifact_sha256": "a" * 64,
                "trust_manifest": str(candidate),
            },
            runtime_identity="r",
            scope_kind="s",
        )
        is None
    )


def test_regular_file_bound_and_uncached_rotation(tmp_path, monkeypatch):
    path = tmp_path / "trust.json"
    raw = json.dumps(empty_manifest()).encode()
    path.write_bytes(raw)
    assert trust._read_bytes(path, len(raw)) == raw
    with pytest.raises(ValueError, match="bound"):
        trust._read_bytes(path, len(raw) - 1)
    with pytest.raises(ValueError, match="regular"):
        trust._read_bytes(tmp_path, 100)
    states = iter([raw, b"not json"])
    monkeypatch.setattr(trust, "_read_bytes", lambda path, limit: next(states))
    assert trust.trusted_manifest()["version"] == 1
    assert "version" not in trust.trusted_manifest()
