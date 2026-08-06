from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import textwrap
from pathlib import Path

import pytest

from apps.chat.evals.offline import runner
from apps.chat.evals.offline.network import NetworkAccessError, deny_network
from apps.chat.evals.offline.runner import (
    REQUIRED_ARTIFACTS,
    normalized_reproducibility_bytes,
    regenerate_paper_table,
    run_component_evaluation,
    run_test_manifest,
    validate_artifacts,
    write_artifacts,
    write_provenance,
)
from apps.chat.evals.run_offline_evidence import main


def test_deny_network_blocks_and_counts_all_socket_entry_points():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    with deny_network() as attempts:
        sock = socket.socket()
        try:
            with pytest.raises(NetworkAccessError):
                sock.connect(("example.invalid", 443))
            with pytest.raises(NetworkAccessError):
                sock.connect_ex(("example.invalid", 443))
            with pytest.raises(NetworkAccessError):
                socket.create_connection(("example.invalid", 443))
        finally:
            sock.close()

    assert attempts.total == 3
    assert [item["operation"] for item in attempts.details] == [
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
    ]
    assert socket.socket.connect is original_connect
    assert socket.socket.connect_ex is original_connect_ex
    assert socket.create_connection is original_create_connection


def test_deny_network_leaves_local_pure_computation_available():
    with deny_network() as attempts:
        assert sum([1, 2, 3]) == 6

    assert attempts.total == 0
    assert attempts.details == []


def _minimal_datasets():
    envelope = {
        "schema_version": "1.1",
        "frozen_at": "2026-08-06T17:00:00Z",
        "provenance": "synthetic_public",
        "sensitivity": "synthetic_public",
        "rubric_version": "1.1",
        "review": {"status": "approved", "record": "review.yaml"},
    }
    routing = {
        **envelope,
        "dataset_id": "routing-v2",
        "cases": [
            {
                "id": "routing-mini-retrieve",
                "stratum": "favorable",
                "input": {
                    "text": "Search the documents for alpha.",
                    "selected_collection_ids": ["public-a"],
                    "prior_tools": [],
                },
                "gold": {
                    "classifier": {
                        "requires_rag": True,
                        "wants_figures": False,
                        "wants_whole_document": False,
                        "is_retry": False,
                        "requires_local_tools": False,
                    },
                    "reason": "explicit_search",
                    "production_action": "retrieve",
                    "expected_query": "Search the documents for alpha.",
                },
            },
            {
                "id": "routing-mini-prompt",
                "stratum": "favorable",
                "input": {
                    "text": "Search the documents for beta.",
                    "selected_collection_ids": [],
                    "prior_tools": [],
                },
                "gold": {
                    "classifier": {
                        "requires_rag": True,
                        "wants_figures": False,
                        "wants_whole_document": False,
                        "is_retry": False,
                        "requires_local_tools": False,
                    },
                    "reason": "explicit_search",
                    "production_action": "prompt_select_collection",
                    "expected_query": "Search the documents for beta.",
                },
            },
            {
                "id": "routing-mini-retry",
                "stratum": "adversarial_boundary",
                "input": {
                    "text": "Please retry.",
                    "selected_collection_ids": ["public-a"],
                    "prior_tools": ["vector_search"],
                    "prior_vector_queries": ["alpha query"],
                },
                "gold": {
                    "classifier": {
                        "requires_rag": True,
                        "wants_figures": False,
                        "wants_whole_document": False,
                        "is_retry": True,
                        "requires_local_tools": False,
                    },
                    "reason": "retry_request",
                    "production_action": "retrieve",
                    "direct_pipeline_action": "skip_normal_tool_loop",
                    "expected_query": "alpha query",
                },
            },
            {
                "id": "routing-mini-skip",
                "stratum": "favorable",
                "input": {
                    "text": "Hello there.",
                    "selected_collection_ids": [],
                    "prior_tools": [],
                },
                "gold": {
                    "classifier": {
                        "requires_rag": False,
                        "wants_figures": False,
                        "wants_whole_document": False,
                        "is_retry": False,
                        "requires_local_tools": False,
                    },
                    "reason": "no_retrieval_needed",
                    "production_action": "skip_normal_tool_loop",
                },
            },
            {
                "id": "routing-mini-local-helper-only",
                "stratum": "unfavorable",
                "input": {
                    "text": "Run the FITS tool on the uploaded file.",
                    "selected_collection_ids": [],
                    "prior_tools": [],
                },
                "gold": {
                    "classifier": {
                        "requires_rag": False,
                        "wants_figures": False,
                        "wants_whole_document": False,
                        "is_retry": False,
                        "requires_local_tools": True,
                    },
                    "reason": "local_tool_request",
                    "production_action": "local_tool_handling",
                },
            },
        ],
    }
    evidence = {
        **envelope,
        "dataset_id": "evidence-v2",
        "cases": [
            {
                "id": "evidence-mini",
                "stratum": "favorable",
                "question": "What is alpha?",
                "answer_target": "Alpha is one.",
                "token_budget": 16,
                "candidates": [
                    {
                        "evidence_id": "e1",
                        "doc_id": "doc-a",
                        "chunk_id": 1,
                        "rank": 1,
                        "text": "Alpha is one.",
                        "citation": "[doc:doc-a chunk:1]",
                        "relevant": True,
                        "estimated_tokens": 3,
                    },
                    {
                        "evidence_id": "e2",
                        "doc_id": "doc-b",
                        "chunk_id": 1,
                        "rank": 2,
                        "text": "Noise only.",
                        "citation": "[doc:doc-b chunk:1]",
                        "relevant": False,
                        "estimated_tokens": 2,
                    },
                ],
                "gold": {
                    "relevant_evidence_ids": ["e1"],
                    "relevant_document_ids": ["doc-a"],
                },
            }
        ],
    }
    memory = {
        **envelope,
        "dataset_id": "memory-v2",
        "cases": [
            {
                "id": "memory-mini-remember",
                "stratum": "favorable",
                "input": {
                    "user_content": "Remember that the synthetic project uses YAML.",
                    "assistant_content": "Okay.",
                },
                "gold": {"normalized_facts": ["The synthetic project uses YAML."]},
            },
            {
                "id": "memory-mini-heuristic",
                "stratum": "favorable",
                "input": {
                    "user_content": "I prefer concise reports.",
                    "assistant_content": "Okay.",
                },
                "gold": {"normalized_facts": ["I prefer concise reports."]},
            },
        ],
    }
    return {"routing": routing, "evidence": evidence, "memory": memory}


def test_component_runner_calls_actual_production_functions_and_restores_env(
    tmp_path, monkeypatch
):
    datasets = _minimal_datasets()
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    for name in datasets:
        (fixture_dir / f"{name}.yaml").write_text(
            "temporary fixture\n", encoding="utf-8"
        )

    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda path, kind: copy.deepcopy(datasets[kind]),
    )
    original_values = {name: os.environ.get(name) for name in runner.CANONICAL_ENV}
    os.environ["RAG_DIRECT_ENABLED"] = "ambient-value"

    wrapped_names = [
        "classify_chat_message",
        "build_retrieval_query",
        "build_evidence_packet",
        "heuristic_facts_from_turn",
        "clean_stable_facts",
        "run_direct_rag_turn",
        "_configure_append_tools",
        "promote_profile_facts_for_turn",
    ]
    originals = {name: getattr(runner, name) for name in wrapped_names}
    calls = dict.fromkeys(wrapped_names, 0)

    for name, original in originals.items():

        def wrapper(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(runner, name, wrapper)

    result = run_component_evaluation(fixture_dir, timing_repeats=2)

    assert all(count > 0 for count in calls.values())
    assert result["network_attempts"]["total"] == 0
    assert result["canonical_env"] == runner.CANONICAL_ENV
    retry = next(r for r in result["routing"] if r["case_id"] == "routing-mini-retry")
    assert retry["actual"]["helper_action"] == "retrieve"
    assert retry["actual"]["direct_action"] == "skip_normal_tool_loop"
    assert retry["actual"]["query"] == "alpha query"
    assert (
        result["memory_fallback"]["explicit_remember"]["branch"] == "explicit_remember"
    )
    assert result["memory_fallback"]["heuristic"]["branch"] == "heuristic"
    assert result["memory_fallback"]["explicit_remember"]["remote_attempt_count"] == 1
    assert result["memory_fallback"]["explicit_remember"]["normalize_calls"] > 0
    assert result["memory_fallback"]["explicit_remember"]["heuristic_calls"] == 0
    assert result["memory_fallback"]["heuristic"]["remote_attempt_count"] == 1
    assert result["memory_fallback"]["heuristic"]["heuristic_calls"] > 0
    assert result["memory_fallback"]["network_failure_latency_seconds"] >= 0
    assert len(result["routing"]) == len(datasets["routing"]["cases"])
    assert all(item["phase"] == "timing" for item in result["timings"])
    assert result["aggregate"]["routing"]["support"] == len(
        datasets["routing"]["cases"]
    )
    local = next(
        record
        for record in result["routing"]
        if record["case_id"] == "routing-mini-local-helper-only"
    )
    assert local["expected"]["direct_action"] is None
    assert local["diagnostics"]["checks"]["direct_action"]["status"] == "not_applicable"
    assert result["aggregate"]["action"]["direct"]["support"] == 4
    direct_labels = result["aggregate"]["action"]["direct"]["by_label"]
    assert direct_labels["retrieve"]["support"] > 0
    assert direct_labels["prompt_select_collection"]["support"] > 0
    assert direct_labels["skip_normal_tool_loop"]["support"] > 0
    assert (
        result["aggregate"]["action"]["direct"]["by_stratum"]["unfavorable"]["support"]
        == 0
    )
    assert result["aggregate"]["action"]["helper"]["by_stratum"]
    assert result["aggregate"]["query"]["by_stratum"]
    evidence_aggregate = result["aggregate"]["evidence"]["aquillm"]
    assert evidence_aggregate["overall"]["relevant_document_coverage"]
    assert evidence_aggregate["by_stratum"]["favorable"]["support"] == 1
    assert evidence_aggregate["overall"]["citation_syntax_validity"]
    assert (
        "image_path_prefix_behavior"
        in result["aggregate"]["evidence"]["paired_comparisons"]
    )
    diversity = result["aggregate"]["evidence"]["paired_comparisons"][
        "distinct_selected_documents"
    ]
    assert diversity["higher_is_better"] is True
    assert diversity["interpretation"] == "descriptive_not_quality"
    evidence_timings = [
        item for item in result["timings"] if item["module"] == "evidence"
    ]
    assert [item["input_size"]["candidate_count"] for item in evidence_timings] == [
        1,
        10,
        100,
    ]
    assert os.environ["RAG_DIRECT_ENABLED"] == "ambient-value"
    for name, value in original_values.items():
        if name == "RAG_DIRECT_ENABLED":
            continue
        assert os.environ.get(name) == value


def test_run_test_manifest_counts_junit_and_represents_blocked_nodes(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mini.py").write_text(
        "def test_passes():\n    assert True\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "test_manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_mini.py::test_passes"
                status: included
                prerequisite: none
                reason: "Pure test."
              - node_id: "tests/test_db.py::test_blocked"
                status: prerequisite_blocked
                prerequisite: postgresql_test_database
                reason: "Database unavailable."
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = run_test_manifest(manifest, tmp_path)

    assert result["summary"] == {
        "collected": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "unavailable": 1,
    }, result["stderr"] + result["stdout"]
    assert [entry["outcome"] for entry in result["entries"]] == [
        "passed",
        "unavailable",
    ]
    assert result["network_scope"] == "component_and_pytest_subprocess"
    assert result["declared_network_policy"] == "no_network"
    assert result["enforced_subprocess_network_denial"] is True
    assert result["subprocess_network_attempts"] == {
        "status": "available",
        "total": 0,
        "details": [],
    }
    assert "test-only" not in result["stdout"] + result["stderr"]


def test_subprocess_environment_is_allowlisted_and_strips_ambient_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("UNLISTED_SERVICE_TOKEN", "ambient-never-leak-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("UNRELATED_SETTING", "private-value")

    env = runner._subprocess_environment(
        Path(runner.__file__).parents[4], tmp_path / "runtime"
    )

    assert "UNLISTED_SERVICE_TOKEN" not in env
    assert "HTTPS_PROXY" not in env
    assert "UNRELATED_SETTING" not in env
    assert env["DJANGO_SETTINGS_MODULE"] == "aquillm.settings"
    assert env["OPENAI_API_KEY"] == "offline-test-only"


def test_subprocess_environment_resolves_dependency_roots_without_site_discovery(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(runner.site, "getusersitepackages", lambda: [])

    env = runner._subprocess_environment(
        Path(runner.__file__).parents[4], tmp_path / "runtime"
    )
    python_paths = set(env["PYTHONPATH"].split(os.pathsep))

    for module_name in ("pytest", "yaml", "django"):
        spec = importlib.util.find_spec(module_name)
        assert spec is not None
        assert spec.submodule_search_locations is not None
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        assert str(package_dir.parent.resolve()) in python_paths


def test_run_test_manifest_times_out_and_terminates_process_tree(
    tmp_path, monkeypatch
):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_slow.py").write_text(
        "def test_slow():\n    assert True\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_slow.py::test_slow"
                status: included
                prerequisite: none
                reason: "Timeout contract."
            """
        ),
        encoding="utf-8",
    )
    terminated = []

    class FakeProcess:
        pid = 321
        returncode = None

        def __init__(self, *_args, **_kwargs):
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise runner.subprocess.TimeoutExpired(["pytest"], timeout)
            self.returncode = -9
            return "partial stdout", "partial stderr"

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        runner,
        "_terminate_process_tree",
        lambda process: terminated.append(process.pid),
    )

    result = run_test_manifest(manifest, tmp_path, timeout_seconds=0.01)

    assert terminated == [321]
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 0.01
    assert result["exit_code"] == 124
    assert result["integrity_failure"] == "pytest_timeout"
    assert result["entries"][0]["outcome"] == "timeout"


def test_run_test_manifest_reports_unavailable_network_audit_as_initialization_failure(
    tmp_path, monkeypatch
):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_startup.py").write_text(
        "def test_startup():\n    assert True\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_startup.py::test_startup"
                status: included
                prerequisite: none
                reason: "Startup failure contract."
            """
        ),
        encoding="utf-8",
    )

    class FailedStartup:
        pid = 456
        returncode = 1

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, timeout=None):
            return "", "No module named pytest"

    monkeypatch.setattr(runner.subprocess, "Popen", FailedStartup)

    result = run_test_manifest(manifest, tmp_path)

    assert (
        result["integrity_failure"]
        == "subprocess_initialization_or_network_audit_failure"
    )
    assert result["subprocess_network_attempts"] == {
        "status": "unavailable",
        "total": None,
        "details": [],
    }
    assert result["configured_subprocess_network_denial"] is True
    assert result["enforced_subprocess_network_denial"] is False


def test_run_test_manifest_uses_ephemeral_home_without_leaking_its_path(
    tmp_path, monkeypatch
):
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_home.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path

            def test_home_is_available():
                assert Path.home().is_dir()
            """
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_home.py::test_home_is_available"
                status: included
                prerequisite: none
                reason: "Synthetic home contract."
            """
        ),
        encoding="utf-8",
    )
    original_popen = runner.subprocess.Popen
    synthetic_paths = {}

    def recording_popen(*args, **kwargs):
        child_env = kwargs["env"]
        for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            path = Path(child_env[name])
            assert path.is_dir()
            synthetic_paths[name] = path
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", recording_popen)

    result = run_test_manifest(manifest, tmp_path)

    assert result["summary"]["collected"] == 1
    assert result["entries"][0]["outcome"] == "passed"
    assert synthetic_paths["HOME"] == synthetic_paths["USERPROFILE"]
    serialized = json.dumps(result)
    for path in synthetic_paths.values():
        assert not path.exists()
        assert str(path) not in serialized


def test_run_test_manifest_enforces_network_denial_inside_pytest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(runner.site, "getusersitepackages", lambda: [])
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_network.py").write_text(
        textwrap.dedent(
            """
            import socket

            def test_network_attempt():
                try:
                    socket.create_connection(("example.invalid", 443))
                except RuntimeError:
                    pass
            """
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_network.py::test_network_attempt"
                status: included
                prerequisite: none
                reason: "Network denial contract."
            """
        ),
        encoding="utf-8",
    )

    result = run_test_manifest(manifest, tmp_path)

    assert result["summary"]["collected"] == 1
    assert result["enforced_subprocess_network_denial"] is True
    assert result["configured_subprocess_network_denial"] is True
    assert result["subprocess_network_attempts"] == {
        "status": "available",
        "total": 1,
        "details": [
            {
                "operation": "socket.create_connection",
                "address": "('example.invalid', 443)",
            }
        ],
    }
    assert result["integrity_failure"] == "subprocess_network_attempt"
    assert result["exit_code"] != 0


def test_run_test_manifest_aggregates_parametrized_nodes_by_identity(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_params.py").write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.parametrize("value", [0, 1])
            def test_param(value):
                if value == 1:
                    pytest.skip("visible parameter skip")

            def test_final():
                assert True
            """
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_params.py::test_param"
                status: included
                prerequisite: none
                reason: "Parametrized."
              - node_id: "tests/test_params.py::test_final"
                status: included
                prerequisite: none
                reason: "Final node."
            """
        ),
        encoding="utf-8",
    )

    result = run_test_manifest(manifest, tmp_path)

    assert result["summary"]["collected"] == 3
    assert [entry["outcome"] for entry in result["entries"]] == [
        "skipped",
        "passed",
    ]
    assert result["entries"][0]["instances"] == {
        "collected": 2,
        "passed": 1,
        "failed": 0,
        "skipped": 1,
        "errors": 0,
    }


def test_test_manifest_rejects_allow_skip_escape_hatch(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_skip.py").write_text(
        "def test_skip():\n    pass\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            schema_version: "1.0"
            entries:
              - node_id: "tests/test_skip.py::test_skip"
                status: included
                prerequisite: none
                reason: "Included tests must pass."
                allow_skip: true
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_skip"):
        run_test_manifest(manifest, tmp_path)


def _artifact_result(timestamp="2026-08-06T18:00:00Z", sample=0.001):
    record = {
        "schema_version": "1.0",
        "module": "routing",
        "case_id": "routing-mini",
        "stratum": "favorable",
        "expected": {"requires_rag": True},
        "actual": {"requires_rag": False},
        "conformant": False,
        "diagnostics": {"reason": "fixed-set miss"},
    }
    aggregate = {
        "schema_version": "1.0",
        "run": {
            "run_id": "mini",
            "timestamp_utc": timestamp,
            "source_commit": "a" * 40,
        },
        "routing": {
            "support": 1,
            "conformance": {
                "numerator": 0,
                "denominator": 1,
                "value": 0.0,
                "status": "ok",
            },
        },
        "action": {"support": 1},
        "query": {"support": 0},
        "evidence": {"support": 0},
        "memory": {"support": 0},
        "tests": {
            "collected": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "unavailable": 1,
        },
        "timing": {
            "routing": {"raw_samples_seconds": [sample], "median_seconds": sample}
        },
        "excluded_claims": ["No generated-answer correctness claim."],
    }
    return {
        "manifest": {
            "schema_version": "1.0",
            "run_id": "mini",
            "timestamp_utc": timestamp,
            "source_commit": "a" * 40,
            "source_dirty": False,
            "fixture_hashes": {"routing.yaml": "b" * 64},
            "code_hashes": {"runner.py": "c" * 64},
            "config_hashes": {"canonical_env": "d" * 64},
            "canonical_env": dict(runner.CANONICAL_ENV),
            "environment": {
                "os": "Windows",
                "processor": "generic",
                "python": "3.13.0",
            },
            "component_network_attempts": {"total": 0, "details": []},
            "test_manifest_hash": "e" * 64,
        },
        "routing": [record],
        "evidence": [],
        "memory": [],
        "timings": [
            {
                "schema_version": "1.0",
                "module": "routing",
                "phase": "timing",
                "raw_samples_seconds": [sample],
            }
        ],
        "tests": {
            "entries": [
                {
                    "node_id": "tests/test_mini.py::test_passes",
                    "status": "included",
                    "outcome": "passed",
                },
                {
                    "node_id": "tests/test_db.py::test_blocked",
                    "status": "prerequisite_blocked",
                    "outcome": "unavailable",
                    "reason": "Database unavailable.",
                },
            ],
            "summary": aggregate["tests"],
            "exit_code": 0,
        },
        "aggregate": aggregate,
    }


def test_artifact_write_is_atomic_complete_immutable_and_valid(tmp_path):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)

    assert set(path.name for path in output.iterdir()) == REQUIRED_ARTIFACTS
    validate_artifacts(output)
    complete = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
    assert set(complete["sha256"]) == REQUIRED_ARTIFACTS - {"COMPLETE"}
    assert (output / "paper-table.md").read_text(
        encoding="utf-8"
    ) == regenerate_paper_table(output / "aggregate.json")
    assert (output / "routing.jsonl").read_bytes().endswith(b"\n")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Routing reason conformance" in report
    assert "passed: 1" in report
    assert "routing" in report and "Median" in report

    with pytest.raises(FileExistsError):
        write_artifacts(_artifact_result(), output)


@pytest.mark.parametrize(
    "private_value",
    [
        "api_key=super-secret-value",
        "AKIAIOSFODNN7EXAMPLE",
        "C:\\Users\\private-person\\source.txt",
        "/home/private-person/source.txt",
    ],
)
def test_artifact_writer_rejects_secret_and_private_path_patterns(
    tmp_path, private_value
):
    result = _artifact_result()
    result["routing"][0]["diagnostics"]["unsafe"] = private_value

    with pytest.raises(ValueError, match="sensitive|private|credential"):
        write_artifacts(result, tmp_path / "unsafe")


def test_artifact_writer_rejects_unknown_inherited_secret_without_echoing_it(
    tmp_path, monkeypatch
):
    secret = "ambient-never-leak-value"
    monkeypatch.setenv("UNLISTED_SERVICE_TOKEN", secret)
    result = _artifact_result()
    result["routing"][0]["diagnostics"]["unsafe"] = f"prefix-{secret}-suffix"

    with pytest.raises(ValueError) as exc_info:
        write_artifacts(result, tmp_path / "unsafe")

    assert "inherited credential" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_artifact_writer_rejects_non_public_fixture_sensitivity(tmp_path):
    result = _artifact_result()
    result["manifest"]["fixture_sensitivity"] = "private"

    with pytest.raises(ValueError, match="synthetic_public"):
        write_artifacts(result, tmp_path / "unsafe")


def test_normalized_reproducibility_and_table_are_timing_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_artifacts(_artifact_result("2026-08-06T18:00:00Z", 0.001), first)
    write_artifacts(_artifact_result("2026-08-06T18:00:01Z", 0.009), second)

    assert normalized_reproducibility_bytes(first) == normalized_reproducibility_bytes(
        second
    )
    assert regenerate_paper_table(first / "aggregate.json") == (
        first / "paper-table.md"
    ).read_text(encoding="utf-8")


def test_write_provenance_records_existing_artifact_commit_and_hashes(tmp_path):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)
    provenance = tmp_path / "provenance.json"

    write_provenance(output, "f" * 40, provenance)

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert payload["evaluated_source_commit"] == "a" * 40
    assert payload["artifact_commit"] == "f" * 40
    assert payload["aggregate_sha256"]
    assert payload["artifact_hashes"]


def test_write_provenance_rejects_tampered_aggregate(tmp_path):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)
    aggregate_path = output / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["routing"]["support"] = 999
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        write_provenance(output, "f" * 40, tmp_path / "provenance.json")


def test_write_provenance_rejects_contradictory_source_commit(tmp_path):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)
    aggregate_path = output / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["run"]["source_commit"] = "b" * 40
    aggregate_path.write_bytes(runner.canonical_json_bytes(aggregate))
    complete_path = output / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["sha256"]["aggregate.json"] = runner.sha256_file(aggregate_path)
    complete_path.write_bytes(runner.canonical_json_bytes(complete))

    with pytest.raises(ValueError, match="source commits contradict"):
        write_provenance(output, "f" * 40, tmp_path / "provenance.json")


def test_validate_rejects_unknown_extra_and_changed_hash(tmp_path):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)
    (output / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_artifacts(output)
    (output / "extra.txt").unlink()
    (output / "routing.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_artifacts(output)


@pytest.mark.parametrize(
    "artifact_name",
    ["routing.csv", "evidence.csv", "memory.csv", "report.md"],
)
def test_validate_rejects_regenerated_artifact_tampering_with_updated_hash(
    tmp_path, artifact_name
):
    output = tmp_path / "run"
    write_artifacts(_artifact_result(), output)
    artifact = output / artifact_name
    artifact.write_text(
        artifact.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    complete_path = output / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["sha256"][artifact_name] = runner.sha256_file(artifact)
    complete_path.write_bytes(runner.canonical_json_bytes(complete))

    with pytest.raises(ValueError, match="does not regenerate"):
        validate_artifacts(output)


def test_csv_and_report_renderers_are_pure_and_deterministic():
    result = _artifact_result()

    first_csv = runner.render_csv(result["routing"])
    first_report = runner.render_report(result["aggregate"])

    assert runner.render_csv(copy.deepcopy(result["routing"])) == first_csv
    assert runner.render_report(copy.deepcopy(result["aggregate"])) == first_report
    assert first_csv.startswith("schema_version,module,case_id")
    assert first_report.startswith("# Preliminary offline component evaluation report")


def test_cli_conformance_misses_succeed_but_integrity_failures_do_not(
    tmp_path, monkeypatch
):
    result = _artifact_result()
    monkeypatch.setattr(
        runner,
        "run_component_evaluation",
        lambda *_args, **_kwargs: copy.deepcopy(result),
    )
    monkeypatch.setattr(
        runner, "run_test_manifest", lambda *_args, **_kwargs: result["tests"]
    )
    monkeypatch.setattr(runner, "_git_source_state", lambda _root: ("a" * 40, False))

    first = tmp_path / "first"
    assert (
        main(
            [
                "run",
                "--fixtures",
                str(tmp_path),
                "--test-manifest",
                str(tmp_path / "manifest.yaml"),
                "--output",
                str(first),
                "--timing-repeats",
                "1",
            ]
        )
        == 0
    )

    failing = copy.deepcopy(result)
    failing["manifest"]["component_network_attempts"] = {
        "total": 1,
        "details": [{"operation": "socket.create_connection", "address": "redacted"}],
    }
    monkeypatch.setattr(
        runner, "run_component_evaluation", lambda *_args, **_kwargs: failing
    )
    assert (
        main(
            [
                "run",
                "--fixtures",
                str(tmp_path),
                "--test-manifest",
                str(tmp_path / "manifest.yaml"),
                "--output",
                str(tmp_path / "second"),
                "--timing-repeats",
                "1",
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    ("test_update", "expected_exit"),
    [
        ({"exit_code": 3}, 1),
        ({"integrity_failure": "pytest_timeout", "timed_out": True}, 1),
        (
            {
                "summary": {
                    "collected": 1,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 1,
                    "errors": 0,
                    "unavailable": 0,
                }
            },
            1,
        ),
        (
            {
                "entries": [
                    {
                        "node_id": "tests/x.py::test_x",
                        "status": "included",
                        "outcome": "missing",
                    }
                ]
            },
            1,
        ),
        (
            {
                "summary": {
                    "collected": 1,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 1,
                    "errors": 0,
                    "unavailable": 0,
                },
                "entries": [
                    {
                        "node_id": "tests/x.py::test_x",
                        "status": "included",
                        "outcome": "skipped",
                        "allow_skip": True,
                    }
                ],
            },
            1,
        ),
    ],
)
def test_cli_rejects_pytest_exit_skip_and_missing_outcomes(
    tmp_path, monkeypatch, test_update, expected_exit
):
    result = _artifact_result()
    tests = copy.deepcopy(result["tests"])
    tests.update(test_update)
    monkeypatch.setattr(runner, "_git_source_state", lambda _root: ("a" * 40, False))
    monkeypatch.setattr(
        runner,
        "run_component_evaluation",
        lambda *_args, **_kwargs: copy.deepcopy(result),
    )
    monkeypatch.setattr(runner, "run_test_manifest", lambda *_args, **_kwargs: tests)

    assert (
        main(
            [
                "run",
                "--fixtures",
                str(tmp_path),
                "--test-manifest",
                str(tmp_path / "manifest.yaml"),
                "--output",
                str(tmp_path / "out"),
                "--timing-repeats",
                "1",
            ]
        )
        == expected_exit
    )


def test_cli_preflights_clean_source_before_execution(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner,
        "_git_source_state",
        lambda _root: calls.append("preflight") or ("a" * 40, True),
    )
    monkeypatch.setattr(
        runner,
        "run_component_evaluation",
        lambda *_args, **_kwargs: calls.append("component"),
    )
    monkeypatch.setattr(
        runner,
        "run_test_manifest",
        lambda *_args, **_kwargs: calls.append("tests"),
    )

    assert (
        main(
            [
                "run",
                "--fixtures",
                str(tmp_path),
                "--test-manifest",
                str(tmp_path / "manifest.yaml"),
                "--output",
                str(tmp_path / "out"),
                "--timing-repeats",
                "1",
            ]
        )
        == 1
    )
    assert calls == ["preflight"]


def test_build_manifest_hashes_all_exercised_code_and_dependencies(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "routing.yaml").write_text("synthetic\n", encoding="utf-8")
    test_manifest = tmp_path / "manifest.yaml"
    test_manifest.write_text("synthetic\n", encoding="utf-8")
    project_root = Path(__file__).parents[3]

    manifest = runner.build_manifest(
        fixture_dir,
        test_manifest,
        project_root,
        2,
        {"total": 0, "details": []},
        source_state=("a" * 40, False),
    )

    assert {
        "apps/chat/consumers/chat_receive.py",
        "aquillm/memory.py",
        "apps/chat/evals/offline/metrics.py",
        "apps/chat/evals/offline/policies.py",
        "apps/chat/evals/offline/schema.py",
        "apps/chat/evals/offline/runner.py",
    }.issubset(manifest["code_hashes"])
    assert {"Django", "PyYAML", "pydantic", "pytest"}.issubset(
        manifest["environment"]["dependencies"]
    )
