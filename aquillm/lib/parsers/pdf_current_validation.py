"""Fixed, bounded current PDF/text correspondence; never historical provenance.

Only validate() is a supported parsing entrypoint. Its POSIX child independently
checks the server's frozen identity before parsing. No storage, Django, models,
alternative policies, text cleanup, or in-process fallback live here.
"""
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Literal, NotRequired, TypedDict


POLICY_KEY = "legacy-upload-PDF-current-validation-v1"
POLICY_VERSION = "1"
MAX_BYTES = 32 * 1024 * 1024
MAX_CANONICAL_BYTES = 8 * 1024 * 1024
MAX_PAGES = 100
MAX_SECONDS = 30
MAX_MEMORY = 1024 * 1024 * 1024
MAX_HEADER = 2048
MAX_REQUEST = MAX_BYTES + MAX_CANONICAL_BYTES + MAX_HEADER + 4
MAX_OUTPUT = 4096

Status = Literal["match", "mismatch", "hashinvalid", "empty", "encrypted",
                 "limits", "timeout", "parsererror", "policychanged", "unsupportedplatform"]
_STATUSES = {"match", "mismatch", "hashinvalid", "empty", "encrypted", "limits",
             "timeout", "parsererror", "policychanged", "unsupportedplatform"}


class PolicyIdentity(TypedDict):
    key: str
    version: str
    implementation_sha256: str
    runtime_sha256: str


class ValidationResult(TypedDict):
    schema: str
    status: Status
    input_bytes: int
    input_sha256: str
    canonical_bytes: int
    canonical_sha256: str
    pages: NotRequired[int]
    projected_bytes: NotRequired[int]
    projected_codepoints: NotRequired[int]
    projected_sha256: NotRequired[str]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze_policy() -> PolicyIdentity:
    """Freeze trusted code/config before capture; persist only this closed identity."""
    import pypdf
    root = Path(pypdf.__file__).resolve().parent
    sources = [(p.relative_to(root).as_posix(), _sha(p.read_bytes()))
               for p in sorted(root.rglob("*.py"))]
    if not sources:
        raise RuntimeError("pypdf source identity unavailable")
    runtime = {"python_version": sys.version, "pypdf_version": pypdf.__version__,
               "pypdf_source_tree_sha256": _sha(_json(sources)),
               "limits": {"pdf_bytes": MAX_BYTES, "canonical_bytes": MAX_CANONICAL_BYTES,
                          "pages": MAX_PAGES, "seconds": MAX_SECONDS, "memory": MAX_MEMORY,
                          "header_bytes": MAX_HEADER, "request_bytes": MAX_REQUEST,
                          "output_bytes": MAX_OUTPUT}}
    return {"key": POLICY_KEY, "version": POLICY_VERSION,
            "implementation_sha256": _sha(_json([(name, _sha(Path(__file__).with_name(name).read_bytes()))
                for name in ("pdf_current_validation.py", "pdf_validation_process.py")])),
            "runtime_sha256": _sha(_json(runtime))}


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _policy_shape(policy):
    return (type(policy) is dict
            and set(policy) == {"key", "version", "implementation_sha256", "runtime_sha256"}
            and type(policy["key"]) is str and policy["key"] == POLICY_KEY
            and type(policy["version"]) is str and policy["version"] == POLICY_VERSION
            and _digest(policy["implementation_sha256"]) and _digest(policy["runtime_sha256"]))


def _base(raw, canonical):
    return {"schema": "pdf-current-validation-result-v1", "input_bytes": len(raw),
            "input_sha256": _sha(raw), "canonical_bytes": len(canonical),
            "canonical_sha256": _sha(canonical)}


def _preflight(base, canonical, stored_sha256):
    if base["input_bytes"] > MAX_BYTES or base["canonical_bytes"] > MAX_CANONICAL_BYTES:
        return "limits"
    if not _digest(stored_sha256) or base["canonical_sha256"] != stored_sha256:
        return "hashinvalid"
    try:
        canonical.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "parsererror"
    return None


class _ProjectionLimit(Exception):
    pass


def _project(reader):
    """Child-only exact policy; bounds reject instead of changing the text."""
    pages = len(reader.pages)
    if pages > MAX_PAGES:
        raise _ProjectionLimit
    parts = []
    size = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            size += len(text.encode("utf-8")) + (2 if parts else 0)
            if size > MAX_CANONICAL_BYTES:
                raise _ProjectionLimit
            parts.append(text)
    return pages, "\n\n".join(parts).strip().encode("utf-8")


def _evaluate(expected_policy, raw, canonical, stored_sha256):
    """Internal child operation; never use directly for captured source bytes."""
    base = _base(raw, canonical)
    status = _preflight(base, canonical, stored_sha256)
    if status:
        return dict(base, status=status)
    try:
        if not _policy_shape(expected_policy) or expected_policy != freeze_policy():
            return dict(base, status="policychanged")
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            return dict(base, status="encrypted")
        pages, projected = _project(reader)
        status = "empty" if not projected else ("match" if projected == canonical else "mismatch")
        return dict(base, status=status, pages=pages, projected_bytes=len(projected),
                    projected_codepoints=len(projected.decode("utf-8")), projected_sha256=_sha(projected))
    except (_ProjectionLimit, MemoryError):
        return dict(base, status="limits")
    except Exception:
        return dict(base, status="parsererror")


def _valid_receipt(child, base):
    if type(child) is not dict or type(child.get("status")) is not str or child["status"] not in _STATUSES:
        return False
    if any(type(child.get(k)) is not type(v) or child.get(k) != v for k, v in base.items()):
        return False
    projection = {"pages", "projected_bytes", "projected_codepoints", "projected_sha256"}
    if child["status"] not in {"match", "mismatch", "empty"}:
        return set(child) == set(base) | {"status"}
    if set(child) != set(base) | projection | {"status"}:
        return False
    if any(type(child[key]) is not int for key in ("pages", "projected_bytes", "projected_codepoints")):
        return False
    pages, size, points = child["pages"], child["projected_bytes"], child["projected_codepoints"]
    if (not 0 <= pages <= MAX_PAGES or not 0 <= points <= size <= MAX_CANONICAL_BYTES
            or size > 4 * points or not _digest(child["projected_sha256"])):
        return False
    if child["status"] == "empty":
        return size == points == 0 and child["projected_sha256"] == _sha(b"")
    if pages == 0 or size == 0:
        return False
    same = size == base["canonical_bytes"] and child["projected_sha256"] == base["canonical_sha256"]
    return same == (child["status"] == "match")


def _request(policy, raw, canonical, stored_sha256):
    header = _json({"policy": policy, "stored_sha256": stored_sha256,
                    "input_bytes": len(raw), "canonical_bytes": len(canonical)})
    if len(header) > MAX_HEADER:
        raise ValueError("request header exceeds bound")
    return len(header).to_bytes(4, "big") + header + raw + canonical


def _load_pump():
    # Exact trusted sibling loading also supports the standalone isolated worker.
    path = Path(__file__).with_name("pdf_validation_process.py")
    spec = importlib.util.spec_from_file_location("pdf_validation_process", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pump


def _pump(process, request, deadline):
    return _load_pump()(process, request, deadline, max_output=MAX_OUTPUT)


def validate(expected_policy: PolicyIdentity, raw: bytes, canonical_utf8: bytes,
             stored_sha256: str, deadline: float) -> ValidationResult:
    """Validate current correspondence in a POSIX worker, with no fallback.

    deadline is an absolute time.monotonic() deadline shared with capture. This
    operation additionally caps itself at 30 seconds. Inputs must be immutable
    bytes captured by trusted server code; a policy or result is not authority.
    """
    if type(raw) is not bytes or type(canonical_utf8) is not bytes:
        raise TypeError("captured input must be immutable bytes")
    base = _base(raw, canonical_utf8)
    status = _preflight(base, canonical_utf8, stored_sha256)
    if status:
        return dict(base, status=status)
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        return dict(base, status="parsererror")
    if deadline <= time.monotonic():
        return dict(base, status="timeout")
    deadline = min(deadline, time.monotonic() + MAX_SECONDS)
    if not _policy_shape(expected_policy):
        return dict(base, status="policychanged")
    if os.name != "posix":
        return dict(base, status="unsupportedplatform")
    try:
        request = _request(expected_policy, raw, canonical_utf8, stored_sha256)
        pump = _load_pump()
        process = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        status, output = pump(process, request, deadline, max_output=MAX_OUTPUT)
        if status:
            return dict(base, status=status)
        if process.returncode in (-signal.SIGXCPU, -signal.SIGALRM):
            return dict(base, status="timeout")
        if process.returncode:
            return dict(base, status="parsererror")
        child = json.loads(output)
        if _valid_receipt(child, base):
            return child
    except Exception:
        pass
    return dict(base, status="parsererror")


def _worker():
    """Self-contained isolated entrypoint; never import the application package."""
    if os.name != "posix":
        raise SystemExit(2)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS, MAX_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    signal.alarm(MAX_SECONDS)
    receipt_fd = os.dup(1)
    try:
        # Suppress Python and native FD output throughout dependency loading and
        # parsing. Only the small serialized result uses the saved descriptor.
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        logging.disable(logging.CRITICAL)
        request = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(request) > MAX_REQUEST or len(request) < 4:
            raise ValueError("invalid request size")
        header_size = int.from_bytes(request[:4], "big")
        if not 0 < header_size <= MAX_HEADER:
            raise ValueError("invalid header size")
        end = 4 + header_size
        header = json.loads(request[4:end])
        if type(header) is not dict or set(header) != {"policy", "stored_sha256", "input_bytes", "canonical_bytes"}:
            raise ValueError("invalid header")
        for key, limit in (("input_bytes", MAX_BYTES), ("canonical_bytes", MAX_CANONICAL_BYTES)):
            if type(header[key]) is not int or not 0 <= header[key] <= limit:
                raise ValueError("invalid byte count")
        raw_end = end + header["input_bytes"]
        if len(request) != raw_end + header["canonical_bytes"]:
            raise ValueError("request length mismatch")
        result = _evaluate(header["policy"], request[end:raw_end], request[raw_end:], header["stored_sha256"])
        encoded = _json(result)
        if len(encoded) > MAX_OUTPUT:
            raise ValueError("receipt exceeds bound")
        os.write(receipt_fd, encoded)
    finally:
        os.close(receipt_fd)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    _worker()
