"""Explicit-only measurement entrypoint; never collected by the ordinary test glob."""
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest
from apps.ingestion.tests.output_support import upload as upload
from apps.chat.evals.ann_measurement_manifest import manifest, digest, WALL_SECONDS
from apps.chat.evals.ann_measurement_observation import database_identity
from apps.chat.evals.ann_measurement_summary import summarize
from .ann_measurement_corpus import build
from .ann_measurement_execution import invocation

pytestmark = pytest.mark.django_db(transaction=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def source_manifest():
    # Container filesystem bytes actually used. Coordinator separately binds full
    # wrapper source manifest, clean Git revision, image and resource inspection.
    root = Path(__file__).resolve().parents[4]
    paths = sorted(root.rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if "__pycache__" not in p.parts}


def test_disposable_filtered_vector_measurement(upload, monkeypatch, settings):
    output = Path("/artifacts/r5-ann") / str(uuid4())
    output.mkdir(parents=True, exist_ok=False)
    print(f"R5_ANN_ARTIFACT_DIRECTORY={output}")
    frozen, records = manifest(), []
    started = monotonic()
    deadline = started + WALL_SECONDS
    failure = None
    write(output / "preregistered.json", frozen)
    write(output / "source-before.json", source_manifest())
    write(output / "runtime.json", {"python": sys.version, "platform": platform.platform(),
        "preregistration_sha256": digest(frozen), "synthetic_provider": frozen["model"],
        "operator_resource_attestation": None, "wrapper_git_binding": "separate required artifact"})
    try:
        before = database_identity()
        write(output / "database-before.json", before)
        assert len(before["indexes"]) == 1 and all(before["indexes"][0][2:4])
        assert "vector_l2_ops" in before["indexes"][0][1] and before["pgvector"]
        corpus = build(upload, monkeypatch, settings, frozen, output, deadline)
        write(output / "observed-corpus.json", corpus.manifest)
        write(output / "database-after-build.json", database_identity())
        # Freeze actual document/chunk/index/profile receipts before any query result.
        write(output / "observed-corpus-binding.json", {"sha256": digest(corpus.manifest),
            "preregistration_sha256": digest(frozen), "status": "frozen_before_measurement"})
        queries = {q["id"]: q["text"] for q in frozen["queries"]}
        with (output / "invocations.jsonl").open("w", encoding="utf-8") as stream:
            for spec in frozen["schedule"]:
                if monotonic() + 15 >= deadline:
                    break
                record = invocation(corpus, spec, queries[spec["query"]])
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                stream.flush()
                if spec["kind"] in ("control", "empty") and record["status"] != "observed":
                    raise RuntimeError("control prerequisite unavailable; measurement not started")
        write(output / "database-after.json", database_identity())
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write(output / "failure.json", failure)
    finally:
        summary = summarize(records, len(frozen["schedule"]))
        summary.update(preparation_or_runner_failure=failure, elapsed_seconds=monotonic() - started,
            runtime_budget="cooperative stop; original producer/turn deadlines unchanged")
        write(output / "summary.json", summary)
        write(output / "source-after.json", source_manifest())
        write(output / "artifact-sha256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir()) if p.is_file()})
    assert failure is None, f"measurement prerequisite failure; retained {output}"
    assert all(r["status"] != "failed" for r in records), f"invariant failure; retained {output}"
    assert (output / "source-before.json").read_bytes() == (output / "source-after.json").read_bytes()
    # A successful wrapper is collection/plumbing only. Unavailable invocations,
    # unattempted scopes, missing plans and absent attestation remain inconclusive.
