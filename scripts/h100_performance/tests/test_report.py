"""Independent fixtures catch mispairing, cherry-picking and optimistic gates."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "report.py"
    if not path.exists():
        pytest.fail("missing serving comparison reporter")
    spec = importlib.util.spec_from_file_location("h100_report", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def rows(role, factor=1.0, blocks=3, context=512, digest="a" * 64):
    result = []
    for block in range(1, blocks + 1):
        for repeat in range(10):
            ttft = (10 + repeat) * factor / 1000
            decode = (20 + repeat) * factor / 1000
            result.append(dict(label=f"{role}-block{block}", prompt_tokens=context,
                               requested_output_tokens=256, output_tokens=256,
                               input_sha256=digest, repeat=repeat, warmup=False,
                               ttft_seconds=ttft, total_seconds=ttft + 255 * decode,
                               aggregate_decode_seconds_per_token=decode,
                               output_sha256="f" * 64, captured_at=f"2026-10-10T00:{block * 2 + (role == 'candidate'):02d}:{repeat:02d}+00:00"))
    return result


def test_excludes_warmups_and_reports_hand_checked_latency_and_throughput():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    baseline.append(dict(warmup=True, ttft_seconds=10000))
    result = module().build_report(baseline, candidate, bootstrap_samples=50, target_contexts=[512])
    group = result["groups"][0]
    assert group["baseline"]["ttft_ms"]["median"] == pytest.approx(14.5)
    assert group["baseline"]["ttft_ms"]["p95"] == pytest.approx(19.0)
    assert group["baseline"]["aggregate_decode_ms_per_token"] == pytest.approx(24.5)
    assert group["baseline"]["output_tokens_per_second"] == pytest.approx(256 / (0.0145 + 255 * 0.0245))
    assert group["paired_blocks"] == 3
    assert group["ttft_improvement_percent"] == pytest.approx(10)
    assert result["gates"]["serving_improvement"]["status"] == "pass"
    assert result["status"] == "incomplete"
    assert result["gates"]["quality"]["status"] == "missing"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "different_input", "short_output", "nonfinite", "nonalternating"])
def test_invalid_captures_cannot_be_reported_as_a_valid_comparison(mutation):
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    if mutation == "duplicate":
        candidate.append(candidate[0].copy())
    elif mutation == "missing":
        candidate.pop()
    elif mutation == "different_input":
        candidate[0]["input_sha256"] = "b" * 64
    elif mutation == "short_output":
        candidate[0]["output_tokens"] = 255
    elif mutation == "nonfinite":
        candidate[0]["ttft_seconds"] = float("nan")
    else:
        for row in candidate:
            row["captured_at"] = row["captured_at"].replace("T00:", "T01:")
    with pytest.raises(ValueError):
        module().build_report(baseline, candidate, bootstrap_samples=10)


def test_two_block_pairs_are_incomplete_even_with_a_large_speedup():
    result = module().build_report(rows("baseline", blocks=2), rows("candidate", 0.5, blocks=2),
                                   bootstrap_samples=30, target_contexts=[512])
    assert result["gates"]["sample_support"]["status"] == "missing"
    assert result["gates"]["serving_improvement"]["status"] == "missing"
    assert result["status"] == "incomplete"


def test_paired_bootstrap_is_deterministic_and_preserves_pair_information():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    first = module().build_report(baseline, candidate, bootstrap_samples=100, seed=19)
    second = module().build_report(baseline, candidate, bootstrap_samples=100, seed=19)
    assert first["groups"][0]["ttft_improvement_ci95_percent"] == second["groups"][0]["ttft_improvement_ci95_percent"]
    lo, hi = first["groups"][0]["ttft_improvement_ci95_percent"]
    assert lo == pytest.approx(10) and hi == pytest.approx(10)
    # A uniformly faster matched pair should remain faster under resampling.
    assert first["groups"][0]["decode_improvement_ci95_percent"][0] > 0


def test_p95_regression_rejects_fast_median_candidate():
    baseline, candidate = rows("baseline"), rows("candidate", 0.8)
    for row in candidate:
        if row["repeat"] == 9:
            row["ttft_seconds"] = 0.1
            row["total_seconds"] = 0.1 + 255 * row["aggregate_decode_seconds_per_token"]
    result = module().build_report(baseline, candidate, bootstrap_samples=100, target_contexts=[512])
    assert result["gates"]["protected_p95"]["status"] == "fail"
    assert result["status"] == "fail"


def test_distinct_contexts_and_output_sizes_are_not_pooled():
    baseline = rows("baseline") + rows("baseline", context=2048, digest="b" * 64)
    candidate = rows("candidate", 0.9) + rows("candidate", 1.1, context=2048, digest="b" * 64)
    for role, destination in (("baseline", baseline), ("candidate", candidate)):
        extra = rows(role, context=512, digest="c" * 64)
        for row in extra:
            row.update(requested_output_tokens=128, output_tokens=128,
                       total_seconds=row["ttft_seconds"] + 127 * row["aggregate_decode_seconds_per_token"])
        destination.extend(extra)
    # Captures within one block may span different shapes at the same timestamp.
    result = module().build_report(baseline, candidate, bootstrap_samples=30)
    assert len(result["groups"]) == 3
    assert result["groups"][0]["ttft_improvement_percent"] == pytest.approx(0)
    assert result["groups"][0]["requested_output_tokens"] == 128
    assert result["groups"][1]["ttft_improvement_percent"] == pytest.approx(10)
    assert result["groups"][2]["ttft_improvement_percent"] == pytest.approx(-10)


def metrics(label, accepted):
    return dict(label=label, before=[
        'vllm:spec_decode_num_draft_tokens{model_name="x"} 100',
        'vllm:spec_decode_num_accepted_tokens{model_name="x"} 80'], after=[
        'vllm:spec_decode_num_draft_tokens{model_name="x"} 200',
        f'vllm:spec_decode_num_accepted_tokens{{model_name="x"}} {80 + accepted}'])


def test_acceptance_uses_counter_deltas_and_rejects_more_than_two_point_drop():
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
        baseline_metrics=[metrics(f"baseline-block{i}", 80) for i in range(1, 4)],
        candidate_metrics=[metrics(f"candidate-block{i}", 77) for i in range(1, 4)])
    assert report["speculation"]["baseline"]["acceptance_rate"] == pytest.approx(0.8)
    assert report["speculation"]["acceptance_change_percentage_points"] == pytest.approx(-3)
    assert report["gates"]["mtp_acceptance"]["status"] == "fail"


def test_counter_reset_is_missing_evidence_instead_of_negative_acceptance():
    reset = metrics("baseline-block1", 80)
    reset["after"][0] = 'vllm:spec_decode_num_draft_tokens{model_name="x"} 1'
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
        baseline_metrics=[reset] + [metrics(f"baseline-block{i}", 80) for i in (2, 3)],
        candidate_metrics=[metrics(f"candidate-block{i}", 80) for i in range(1, 4)])
    assert report["gates"]["mtp_acceptance"]["status"] == "missing"


def quality(role):
    return [dict(id=f"{category}-{i}", label=role, passed=True, message={"content": "ok"})
            for category in ("number", "tool", "multiturn", "reasoning") for i in range(8)]


def test_lost_quality_pass_fails_and_missing_cases_do_not_pass():
    baseline, candidate = quality("baseline"), quality("candidate")
    candidate[0]["passed"] = False
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
                                   baseline_quality=baseline, candidate_quality=candidate)
    assert report["gates"]["quality"]["status"] == "fail"
    assert report["quality"]["lost_passes"] == ["number-0"]
    candidate[0]["passed"] = True
    candidate.pop()
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
                                   baseline_quality=baseline, candidate_quality=candidate)
    assert report["gates"]["quality"]["status"] == "missing"


def test_absent_candidate_is_missing_evidence():
    result = module().build_report(rows("baseline"), [], bootstrap_samples=10)
    assert result["status"] == "incomplete" and result["groups"] == []


def test_real_prometheus_total_counters_ignore_created_and_per_position_double_counting():
    captures = []
    for role in ("baseline", "candidate"):
        result = []
        for block in range(1, 4):
            snapshot = metrics(f"{role}-block{block}", 80)
            for phase in ("before", "after"):
                snapshot[phase] = [line.replace("tokens{", "tokens_total{") for line in snapshot[phase]]
                snapshot[phase].append('vllm:spec_decode_num_draft_tokens_created{model_name="x"} 1700000000')
                snapshot[phase].append('vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 20')
            result.append(snapshot)
        captures.append(result)
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
                                  baseline_metrics=captures[0], candidate_metrics=captures[1])
    assert report["speculation"]["baseline"]["draft_tokens"] == 300
    assert report["speculation"]["baseline"]["accepted_tokens"] == 240
    assert report["gates"]["mtp_acceptance"]["status"] == "pass"


def test_metrics_missing_one_serving_block_cannot_pass_acceptance_gate():
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
        baseline_metrics=[metrics("baseline-block1", 80)], candidate_metrics=[metrics("candidate-block1", 80)])
    assert report["gates"]["mtp_acceptance"]["status"] == "missing"


def test_discovery_capture_has_missing_completion_and_strict_quality_qualification():
    report = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
                                   baseline_quality=quality("baseline"), candidate_quality=quality("candidate"))
    assert report["gates"]["stream_completion"]["status"] == "missing"
    assert report["gates"]["quality_oracle"]["status"] == "missing"


def test_incomplete_error_attempt_is_excluded_from_latency_but_blocks_promotion():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    for row in baseline + candidate:
        row.update(complete=True, error=None, stream_done=True, finish_reason="length",
                   usage={"completion_tokens": 256})
    candidate[-1].update(complete=False, error="stream ended without DONE", stream_done=False,
                         output_tokens=None, ttft_seconds=None, aggregate_decode_seconds_per_token=None)
    result = module().build_report(baseline, candidate, bootstrap_samples=10, target_contexts=[512])
    assert result["groups"][0]["candidate"]["requests"] == 29
    assert result["candidate_errors"] == 1
    assert result["gates"]["stream_completion"]["status"] == "fail"
    assert result["gates"]["sample_support"]["status"] == "missing"
    assert result["gates"]["errors"]["status"] == "fail"


def test_all_supplied_qualified_evidence_can_pass_and_exact_two_point_drop_is_allowed():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    for row in baseline + candidate:
        row.update(complete=True, error=None, stream_done=True, finish_reason="length",
                   usage={"completion_tokens": 256})
    base_quality, new_quality = quality("baseline"), quality("candidate")
    for row in base_quality + new_quality:
        row.update(oracle="exact-v1", complete=True, error=None, finish_reason="stop",
                   input_sha256="d" * 64, case_sha256="e" * 64)
    report = module().build_report(baseline, candidate, bootstrap_samples=20, target_contexts=[512],
        baseline_metrics=[metrics(f"baseline-block{i}", 80) for i in range(1, 4)],
        candidate_metrics=[metrics(f"candidate-block{i}", 78) for i in range(1, 4)],
        baseline_quality=base_quality, candidate_quality=new_quality,
        evidence={"target_kernel_median_reduction_percent": 10, "activation_verified": True,
                  "runtime_identity_frozen": True, "numerical_graph_checks_passed": True,
                  "memory_gate_passed": True, "application_replay_passed": True})
    assert report["gates"]["mtp_acceptance"]["status"] == "pass"
    assert report["status"] == "pass"


@pytest.mark.parametrize("malformed", ["serving", "metrics"])
def test_cli_emits_invalid_report_instead_of_crashing_on_wrong_json_shapes(tmp_path, malformed):
    source, output = tmp_path / "rows.jsonl", tmp_path / "report.json"
    source.write_text("[]\n" if malformed == "serving" else
                      "\n".join(json.dumps(row) for row in rows("baseline")))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "report.py"),
               "--baseline", str(source), "--output", str(output)]
    if malformed == "metrics":
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text("[]")
        command.extend(["--baseline-metrics", str(metrics_file)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())["status"] == "invalid"
    assert "Traceback" not in result.stderr


def test_candidate_only_strict_quality_rows_remain_missing_evidence():
    candidate = quality("candidate")
    for row in candidate:
        row.update(oracle="exact-v1", complete=True, error=None, finish_reason="stop",
                   input_sha256="d" * 64, case_sha256="e" * 64)
    result = module().build_report(rows("baseline"), rows("candidate", 0.9), bootstrap_samples=10,
                                   candidate_quality=candidate)
    assert result["gates"]["quality_oracle"]["status"] == "missing"


def qualified_report(baseline, candidate, **evidence):
    for row in baseline + candidate:
        row.update(complete=True, error=None, stream_done=True, finish_reason="length",
                   usage={"completion_tokens": 256})
    base_quality, new_quality = quality("baseline"), quality("candidate")
    for row in base_quality + new_quality:
        row.update(oracle="exact-v1", complete=True, error=None, finish_reason="stop",
                   input_sha256="d" * 64, case_sha256="e" * 64)
    return module().build_report(baseline, candidate, bootstrap_samples=10, target_contexts=[512],
        baseline_metrics=[metrics(f"baseline-block{i}", 80) for i in range(1, 4)],
        candidate_metrics=[metrics(f"candidate-block{i}", 80) for i in range(1, 4)],
        baseline_quality=base_quality, candidate_quality=new_quality,
        evidence=dict(target_kernel_median_reduction_percent=10, activation_verified=True,
                      runtime_identity_frozen=True, numerical_graph_checks_passed=True,
                      memory_gate_passed=True, application_replay_passed=True, **evidence))


def test_both_missing_output_hashes_cannot_pass_even_with_other_qualified_evidence():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    for row in baseline + candidate:
        row.pop("output_sha256")
    result = qualified_report(baseline, candidate, greedy_changes_reviewed=True)
    assert result["status"] == "incomplete"
    assert result["gates"]["greedy_output_review"]["status"] == "missing"
    assert result["gates"]["greedy_evidence"]["status"] == "missing"


@pytest.mark.parametrize("invalid", [None, "same", "a" * 63, "g" * 64, 42])
def test_malformed_provided_output_hash_is_invalid(invalid):
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    for row in baseline + candidate:
        row["output_sha256"] = "f" * 64
    candidate[0]["output_sha256"] = invalid
    with pytest.raises(ValueError, match="output_sha256"):
        qualified_report(baseline, candidate)


def test_different_valid_greedy_hashes_require_explicit_review():
    baseline, candidate = rows("baseline"), rows("candidate", 0.9)
    for row in baseline:
        row["output_sha256"] = "a" * 64
    for row in candidate:
        row["output_sha256"] = "b" * 64
    result = qualified_report(baseline, candidate)
    assert result["gates"]["greedy_evidence"]["status"] == "pass"
    assert result["gates"]["greedy_output_review"]["status"] == "missing"
    assert result["groups"][0]["changed_greedy_output_pairs"] == 30
    assert qualified_report(baseline, candidate, greedy_changes_reviewed=True)["status"] == "pass"


def test_cli_malformed_output_hash_emits_invalid_report(tmp_path):
    source, output = tmp_path / "rows.jsonl", tmp_path / "report.json"
    capture = rows("baseline")
    capture[0]["output_sha256"] = "not-a-digest"
    source.write_text("\n".join(json.dumps(row) for row in capture))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "report.py"),
        "--baseline", str(source), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())["status"] == "invalid"
