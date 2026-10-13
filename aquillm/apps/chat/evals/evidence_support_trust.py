"""Administrator enrollment, versioned separately from evaluated source code.

Deploy this fixed file read-only, owned/writable only by the administrator, never
by the application or corpus uploader. No CLI, candidate path or cache can enroll
reviews. Absence/malformed configuration is an empty authority on every use.
"""

import hashlib
import json
import re
import stat
from pathlib import Path

TRUST_PATH = Path("/etc/aquillm/evidence-review-trust.json")
MAX_TRUST_BYTES = 262144
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
PIN_FIELDS = {"reviewer", "provenance_sha256", "criteria_sha256", "subjects_sha256"}


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate review key")
        value[key] = item
    return value


def strict_json(raw):
    def reject(_):
        raise ValueError("non-finite review number")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)


def _read_bytes(path, limit):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("review file must be regular")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("review artifact exceeds bound")
    return raw


def parse_manifest(raw):
    if len(raw) > MAX_TRUST_BYTES:
        raise ValueError("review manifest exceeds bound")
    manifest = strict_json(raw)
    if (
        set(manifest) != {"schema", "version", "calibration", "activation"}
        or manifest["schema"] != "support-review-trust-v1"
        or type(manifest["version"]) is not int
        or manifest["version"] < 1
    ):
        raise ValueError("invalid review manifest schema")
    for kind in ("calibration", "activation"):
        if not isinstance(manifest[kind], dict):
            raise ValueError("invalid review pin registry")
        for sha, pin in manifest[kind].items():
            fields = PIN_FIELDS | (
                {"revision", "policy_sha256"} if kind == "activation" else set()
            )
            if (
                not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not isinstance(pin, dict)
                or set(pin) != fields
                or not isinstance(pin["reviewer"], str)
                or not pin["reviewer"].strip()
                or any(
                    not isinstance(pin[k], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", pin[k])
                    for k in fields
                    if k.endswith("sha256")
                )
                or (
                    kind == "activation"
                    and not re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
                )
            ):
                raise ValueError("invalid review pin")
    if set(manifest["calibration"]) & set(manifest["activation"]):
        raise ValueError("ambiguous review artifact role")
    return manifest


def trusted_manifest():
    try:
        return parse_manifest(_read_bytes(TRUST_PATH, MAX_TRUST_BYTES))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {"calibration": {}, "activation": {}}


def trusted_artifact(attachment, kind):
    sha = attachment["artifact_sha256"]
    pin = trusted_manifest()[kind].get(sha)
    if not pin:
        raise ValueError("support review not independently enrolled")
    raw = _read_bytes(Path(attachment["artifact"]), MAX_ARTIFACT_BYTES)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("support review artifact changed")
    record = strict_json(raw)
    provenance = _read_bytes(Path(record["provenance_artifact"]), MAX_ARTIFACT_BYTES)
    if (
        record["reviewer"] != pin["reviewer"]
        or record["provenance_sha256"] != pin["provenance_sha256"]
        or hashlib.sha256(provenance).hexdigest() != pin["provenance_sha256"]
    ):
        raise ValueError("independent reviewer provenance changed")
    return record, pin
