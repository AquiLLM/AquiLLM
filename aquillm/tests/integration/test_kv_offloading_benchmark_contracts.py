"""Comparison/evidence contracts operate on real loopback benchmark results."""
import copy
import json
import subprocess
import sys

import pytest

from .test_kv_offloading_benchmark import SCRIPT, benchmark, inputs, run, server


def comparable_results():
    with server() as (url, _):
        resident = run(url)
    candidate = copy.deepcopy(resident)
    candidate["profile"]["storage_mode"] = "mooncake"
    for row in candidate["requests"]:
        row["elapsed_seconds"] *= 1.04
        row["first_token_latency_seconds"] *= 1.04
    return candidate, resident


def test_latency_and_rate_targets_are_independent_and_per_request():
    candidate, resident = comparable_results()
    verdict = benchmark().compare_baseline(candidate, resident)
    assert verdict["status"] == "pass"
    assert verdict["preferred_5_percent"] is True
    assert all(row["total_latency_ratio"] == pytest.approx(1.04) for row in verdict["requests"])
    candidate["requests"][1]["elapsed_seconds"] = resident["requests"][1]["elapsed_seconds"] * 1.11
    verdict = benchmark().compare_baseline(candidate, resident)
    assert verdict["status"] == "fail"
    assert verdict["requests"][0]["status"] == "pass"
    assert verdict["requests"][1]["status"] == "fail"


@pytest.mark.parametrize("change", ["missing_identity", "hardware", "runtime", "revision", "mtp", "model",
                                       "quantization", "genesis", "weights", "kv", "n", "t", "workload",
                                       "prompt", "output", "serial", "not_resident", "error", "missing_rate"])
def test_incompatible_or_incomplete_baseline_comparison_is_unavailable(change):
    candidate, resident = comparable_results()
    if change == "missing_identity":
        del resident["profile"]["identity"]["runtime"]
    elif change in {"hardware", "runtime", "revision", "mtp", "model", "quantization", "genesis", "weights"}:
        resident["profile"]["identity"][change] = "different"
    elif change == "kv":
        resident["profile"]["kv_dtype"] = "auto"
    elif change in {"n", "t"}:
        resident["configured_" + change] += 1
    elif change == "workload":
        resident["workload_sha256"] = "different"
    elif change == "prompt":
        resident["requests"][0]["prompt_sha256"] = "different"
    elif change == "output":
        resident["requests"][0]["requested_output_tokens"] += 1
    elif change == "serial":
        resident["client_overlap"]["max_requests"] = 1
    elif change == "not_resident":
        resident["profile"]["storage_mode"] = "mooncake"
    elif change == "error":
        resident["requests"][0]["error"] = "server_error"
    else:
        resident["requests"][0]["accepted_tokens_per_second"] = None
    assert benchmark().compare_baseline(candidate, resident)["status"] == "unavailable"


def test_missing_profile_identity_cannot_pass_measured_targets():
    profile, fixture = inputs()
    del profile["identity"]
    with server() as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8)
    assert result["rate_target"]["status"] == "unavailable"
    assert result["capacity_target"]["status"] == "unavailable"
    assert result["requests"][0]["accepted_tokens_per_second"] > 0


def test_run_correlated_trace_distinguishes_active_paging_from_prefix_restores():
    candidate, _ = comparable_results()
    rows = candidate["requests"]
    start = max(row["started_seconds"] for row in rows)
    end = min(row["ended_seconds"] for row in rows)
    metrics = dict(run_id=candidate["run_id"], source="fixture trace",
                   scheduler_intervals=[dict(start_seconds=start, end_seconds=end, phase="decode",
                                             request_ids=[row["request_id"] for row in rows])],
                   transfers=[dict(request_id=rows[0]["request_id"], at_seconds=(start + end) / 2,
                                   purpose="prefix_restore", direction="h2d", bytes=4096)])
    evidence = benchmark().evidence_summary(candidate, metrics)
    assert evidence["active_decoders"]["status"] == "supported-by-supplied-trace"
    assert evidence["active_paging"]["status"] == "unverified"
    metrics["transfers"][0]["purpose"] = "active_kv"
    evidence = benchmark().evidence_summary(candidate, metrics)
    assert evidence["active_paging"]["status"] == "supported-by-supplied-trace"
    assert evidence["active_paging"]["transfer_bytes"] == 4096
    assert evidence["provenance"] == "operator-supplied-unattested"


@pytest.mark.parametrize("metrics", [{"running": 8}, {"run_id": "other", "active_paging": True},
                                      {"run_id": "match", "scheduler_intervals": [{"request_ids": ["other"]}]}])
def test_gauges_flags_and_wrong_run_traces_are_unverified(metrics):
    candidate, _ = comparable_results()
    candidate["run_id"] = "match"
    evidence = benchmark().evidence_summary(candidate, metrics)
    assert evidence["active_decoders"]["status"] == "unverified"
    assert evidence["active_paging"]["status"] == "unverified"


def test_request_timeout_is_bounded_and_yields_no_rate():
    profile, fixture = inputs()
    with server(delay=0.3) as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8, timeout=0.05)
    assert all(row["error"] == "transport_or_timeout_error" for row in result["requests"])
    assert all(row["elapsed_seconds"] < 0.2 for row in result["requests"])
    assert all(row["accepted_tokens_per_second"] is None for row in result["requests"])


def test_unrelated_profile_fields_do_not_leak_into_report():
    profile, fixture = inputs()
    profile["api_key"] = "secret-that-must-not-appear"
    with server() as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8)
    assert "secret-that-must-not-appear" not in json.dumps(result)


def test_analysis_cli_adds_historical_evidence_without_another_request(tmp_path):
    candidate, resident = comparable_results()
    paths = [tmp_path / name for name in ("run.json", "resident.json", "trace.json", "analysis.json")]
    for path, value in zip(paths, (candidate, resident, {"run_id": candidate["run_id"], "running": 8})):
        path.write_text(json.dumps(value))
    completed = subprocess.run([sys.executable, str(SCRIPT), "--analyze", str(paths[0]),
                                "--baseline", str(paths[1]), "--metrics", str(paths[2]),
                                "--output", str(paths[3])], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(paths[3].read_text())
    assert result["baseline_comparison"]["status"] == "pass"
    assert result["evidence"]["active_decoders"]["status"] == "unverified"
    assert result["requests"] == candidate["requests"]


def test_first_token_latency_cannot_hide_behind_total_latency():
    candidate, resident = comparable_results()
    candidate["requests"][0]["first_token_latency_seconds"] = resident["requests"][0]["first_token_latency_seconds"] * 1.2
    assert benchmark().compare_baseline(candidate, resident)["status"] == "fail"


def test_duplicate_scheduler_ids_cannot_prove_n_decoders():
    candidate, _ = comparable_results()
    row = candidate["requests"][0]
    metrics = dict(run_id=candidate["run_id"], scheduler_intervals=[dict(
        phase="decode", request_ids=[row["request_id"]] * 2,
        start_seconds=row["started_seconds"], end_seconds=row["ended_seconds"])])
    assert benchmark().evidence_summary(candidate, metrics)["active_decoders"]["status"] == "unverified"


def test_served_alias_preserves_underlying_tokenizer_model_identity():
    profile, fixture = inputs()
    with server() as (url, captured):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8,
                                           served_model="fixture-alias")
    assert all(payload["model"] == "fixture-alias" for payload, _ in captured)
    assert result["model"] == "fixture-model"
    assert result["served_model"] == "fixture-alias"


def test_control_characters_in_run_id_are_rejected_before_network():
    profile, fixture = inputs()
    with server() as (url, captured):
        with pytest.raises(ValueError):
            benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8, run_id="bad\r\nheader")
        assert captured == []


def test_profile_identity_model_must_match_request_model_in_analysis():
    candidate, resident = comparable_results()
    for result in (candidate, resident):
        result["profile"]["identity"]["model"] = "different-underlying-model"
    assert benchmark().compare_baseline(candidate, resident)["status"] == "unavailable"


def test_profile_capacity_must_match_result_capacity_in_analysis():
    candidate, resident = comparable_results()
    resident["profile"]["context_tokens"] = 262144
    assert benchmark().compare_baseline(candidate, resident)["status"] == "unavailable"


@pytest.mark.parametrize("field,value", [("mtp", "off"), ("quantization", "fp8")])
def test_unapproved_model_settings_do_not_pass_targets(field, value):
    profile, fixture = inputs()
    profile["identity"][field] = value
    with server() as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8)
    assert result["rate_target"]["status"] == "unavailable"


def test_malformed_analysis_returns_redacted_input_error(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "complete"}))
    completed = subprocess.run([sys.executable, str(SCRIPT), "--analyze", str(path),
                                "--output", str(tmp_path / "out.json")], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 64
    assert "Traceback" not in completed.stderr


def test_dribbling_headers_cannot_extend_request_deadline():
    profile, fixture = inputs()
    with server("slow_headers") as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8, timeout=0.05)
    assert all(row["error"] == "transport_or_timeout_error" for row in result["requests"])
    assert all(row["elapsed_seconds"] < 0.2 for row in result["requests"])
