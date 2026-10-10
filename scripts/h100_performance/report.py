"""CPU-only paired serving report; missing evidence never becomes a pass.

Example: python report.py --baseline baseline-block*.jsonl --candidate
candidate-block*.jsonl --target-context 32768 --output comparison.json
Shell globs must be expanded by the caller. Each role uses labels ending in
blockN; matching N identifies one AB pair. Default support is three AB pairs
with ten non-warmup requests per frozen input in every block. Bootstrap uses
paired block resampling, then paired repeats within each selected block.
Output throughput includes TTFT. Decode timing is aggregate, not stream-chunk
token timestamps. This tool does not change configuration or contact a server.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import statistics


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[math.ceil(position)] * fraction


def _number(row, key, *, positive=False):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"invalid finite number: {key}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"invalid positive timing: {key}")
    return value


def _failed(row):
    return row.get("complete") is False or bool(row.get("error")) or row.get("stream_done") is False


def _completion_gate(rows):
    measured = [row for row in rows if row.get("warmup") is False]
    if any(_failed(row) for row in measured):
        return _gate("fail", "incomplete/error attempts excluded from latency and counted as failures")
    fields = ("complete", "error", "finish_reason", "stream_done")
    if not measured or any(any(field not in row for field in fields) for row in measured):
        return _gate("missing", "discovery captures lack stream completion/error qualification")
    good = all(row["complete"] is True and row["error"] is None and row["stream_done"] is True
               and row["finish_reason"] == "length" and
               (row.get("usage") or {}).get("completion_tokens") == row["requested_output_tokens"]
               for row in measured)
    return _gate("pass" if good else "fail", "requires DONE, terminal length, correct usage and explicit complete/error")


def _capture(rows, expected_repeats):
    groups = defaultdict(lambda: defaultdict(dict))
    intervals = defaultdict(list)
    for row in rows:
        if row.get("warmup") is True:
            continue
        if row.get("warmup") is not False:
            raise ValueError("warmup must be an explicit boolean")
        match = re.search(r"(?:^|-)block([1-9][0-9]*)$", str(row.get("label", "")))
        if not match:
            raise ValueError("serving label must end with -blockN")
        block = int(match[1])
        digest = row.get("input_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("missing frozen input_sha256")
        context, output = row.get("prompt_tokens"), row.get("requested_output_tokens")
        if any(type(value) is not int or value <= 0 for value in (context, output)) or output <= 1:
            raise ValueError("invalid context or requested output size")
        if not _failed(row) and (type(row.get("output_tokens")) is not int or row["output_tokens"] != output):
            raise ValueError("incomplete request: output_tokens differs from frozen requested output")
        repeat = row.get("repeat")
        if type(repeat) is not int or not 0 <= repeat < expected_repeats:
            raise ValueError("repeat outside expected range")
        key = (digest, context, output)
        if repeat in groups[key][block]:
            raise ValueError(f"duplicate repeat {repeat} in block {block}")
        if not _failed(row):
            if "output_sha256" in row and (not isinstance(row["output_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["output_sha256"])):
                raise ValueError("invalid output_sha256: expected lowercase SHA256 digest")
            ttft = _number(row, "ttft_seconds", positive=True)
            total = _number(row, "total_seconds", positive=True)
            decode = _number(row, "aggregate_decode_seconds_per_token", positive=True)
            if total <= ttft or not math.isclose(decode, (total - ttft) / (output - 1), rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError("decode time does not agree with completion, TTFT and token count")
        try:
            captured = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError, AttributeError) as error:
            raise ValueError("invalid captured_at timestamp") from error
        if captured.tzinfo is None:
            raise ValueError("captured_at must have a timezone")
        groups[key][block][repeat] = row
        intervals[block].append(captured)
    expected = set(range(expected_repeats))
    all_blocks = set(intervals)
    for key, blocks in groups.items():
        if set(blocks) != all_blocks:
            raise ValueError(f"missing shape in a block: {key}")
        for block, repeats in blocks.items():
            if set(repeats) != expected:
                raise ValueError(f"missing repeats in block {block}: {sorted(expected - set(repeats))}")
    return groups, {block: (min(times), max(times)) for block, times in intervals.items()}


def _stats(rows):
    if not rows:
        return {"requests": 0, "ttft_ms": {"median": None, "p95": None},
                "aggregate_decode_ms_per_token": None, "decode_ms_per_token": {"median": None, "p95": None},
                "output_tokens_per_second": None}
    ttft = [row["ttft_seconds"] * 1000 for row in rows]
    decode = [row["aggregate_decode_seconds_per_token"] * 1000 for row in rows]
    return {
        "requests": len(rows), "ttft_ms": {"median": statistics.median(ttft), "p95": percentile(ttft, 0.95)},
        "aggregate_decode_ms_per_token": sum(row["total_seconds"] - row["ttft_seconds"] for row in rows)
        / sum(row["output_tokens"] - 1 for row in rows) * 1000,
        "decode_ms_per_token": {"median": statistics.median(decode), "p95": percentile(decode, 0.95)},
        "output_tokens_per_second": sum(row["output_tokens"] for row in rows) / sum(row["total_seconds"] for row in rows),
    }


def _improvement(baseline, candidate):
    if baseline is None or candidate is None:
        return None
    return (1 - candidate / baseline) * 100


def _bootstrap(baseline, candidate, samples, seed):
    generator = random.Random(seed)
    valid = {block: [repeat for repeat, row in repeats.items()
                     if not _failed(row) and not _failed(candidate[block][repeat])]
             for block, repeats in baseline.items()}
    blocks = [block for block in sorted(baseline) if valid[block]]
    if not blocks:
        return [None, None], [None, None]
    ttft_changes, decode_changes = [], []
    for _ in range(samples):
        base_rows, candidate_rows = [], []
        for block in generator.choices(blocks, k=len(blocks)):
            for repeat in generator.choices(valid[block], k=len(valid[block])):
                base_rows.append(baseline[block][repeat])
                candidate_rows.append(candidate[block][repeat])
        base, new = _stats(base_rows), _stats(candidate_rows)
        ttft_changes.append(_improvement(base["ttft_ms"]["median"], new["ttft_ms"]["median"]))
        decode_changes.append(_improvement(base["aggregate_decode_ms_per_token"], new["aggregate_decode_ms_per_token"]))
    return ([percentile(ttft_changes, q) for q in (0.025, 0.975)],
            [percentile(decode_changes, q) for q in (0.025, 0.975)])


def _gate(status, reason):
    return {"status": status, "reason": reason}


def _metrics_delta(snapshots, labels):
    if not snapshots:
        return {"status": "missing", "reason": "speculation before/after counters absent"}
    if {snapshot.get("label") for snapshot in snapshots} != labels:
        return {"status": "missing", "reason": "metrics snapshots must cover every serving block label"}
    totals = defaultdict(float)
    seen_labels = set()
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+0-9.eE]+)(?:\s+\S+)?$")
    for snapshot in snapshots:
        label = snapshot.get("label")
        if not label or label in seen_labels:
            return {"status": "missing", "reason": "duplicate/missing metrics snapshot labels"}
        seen_labels.add(label)
        parsed = []
        for phase in ("before", "after"):
            series = {}
            for line in snapshot.get(phase, []):
                match = pattern.fullmatch(line.strip())
                if not match:
                    return {"status": "missing", "reason": "malformed speculation metric"}
                value = float(match[3])
                if match[1].endswith("_created"):
                    continue
                identity = (match[1].removesuffix("_total"), match[2] or "")
                if not math.isfinite(value) or value < 0 or identity in series:
                    return {"status": "missing", "reason": "invalid/duplicate metric series"}
                series[identity] = value
            parsed.append(series)
        before, after = parsed
        if not before or set(before) != set(after):
            return {"status": "missing", "reason": "metric series missing before or after"}
        for identity, value in after.items():
            delta = value - before[identity]
            if delta < 0:
                return {"status": "missing", "reason": "counter reset during captured interval"}
            totals[identity[0]] += delta
    draft = sum(value for name, value in totals.items() if name.endswith("spec_decode_num_draft_tokens"))
    accepted = sum(value for name, value in totals.items() if name.endswith("spec_decode_num_accepted_tokens"))
    if draft <= 0 or not any(name.endswith("spec_decode_num_accepted_tokens") for name in totals) or accepted > draft:
        return {"status": "missing", "reason": "nonzero draft and valid accepted-token counters required", "deltas": dict(totals)}
    return {"status": "measured", "deltas": dict(totals), "draft_tokens": draft,
            "accepted_tokens": accepted, "acceptance_rate": accepted / draft,
            "interval_includes_warmups": True}


def _quality(baseline, candidate):
    expected = {f"{kind}-{index}" for kind in ("number", "tool", "multiturn", "reasoning") for index in range(8)}
    indexed = []
    for rows in (baseline or [], candidate or []):
        values = {}
        for row in rows:
            identity = row.get("id")
            if identity in values or type(row.get("passed")) is not bool:
                return {"status": "missing", "reason": "duplicate quality ID or invalid pass result"}
            values[identity] = row
        indexed.append(values)
    base, new = indexed
    if set(base) != expected or set(new) != expected:
        return {"status": "missing", "reason": "all frozen 32 quality cases required",
                "baseline_missing": sorted(expected - set(base)), "candidate_missing": sorted(expected - set(new))}
    lost = sorted(identity for identity in expected if base[identity]["passed"] and not new[identity]["passed"])
    changed = sorted(identity for identity in expected if base[identity].get("message") != new[identity].get("message"))
    return {"status": "fail" if lost else "pass", "reason": "preserve every baseline pass",
            "lost_passes": lost, "changed_outputs": changed,
            "baseline_passes": sum(row["passed"] for row in base.values()),
            "candidate_passes": sum(row["passed"] for row in new.values())}


def _quality_oracle(baseline, candidate):
    captures = (baseline or []) + (candidate or [])
    if not baseline or not candidate or any(not isinstance(row.get("id"), str)
                           or row.get("oracle") != "exact-v1" or "complete" not in row or "error" not in row
                           or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("input_sha256", "")))
                           or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("case_sha256", ""))) for row in captures):
        return _gate("missing", "requires exact-v1 oracle, frozen input/case hashes and completion/error fields")
    base = {row["id"]: row for row in baseline}
    new = {row["id"]: row for row in candidate}
    if set(base) != set(new):
        return _gate("missing", "quality cases differ")
    if any(base[key][field] != new[key][field] for key in base for field in ("input_sha256", "case_sha256")):
        return _gate("fail", "quality frozen inputs/cases differ")
    good = all(row["complete"] is True and row["error"] is None and row.get("finish_reason") in ("stop", "tool_calls") for row in captures)
    return _gate("pass" if good else "fail", "strict completed quality captures required")


def build_report(baseline_rows, candidate_rows, *, expected_repeats=10, bootstrap_samples=2000,
                 seed=17, target_contexts=None, baseline_metrics=None, candidate_metrics=None,
                 baseline_quality=None, candidate_quality=None, evidence=None):
    if type(expected_repeats) is not int or expected_repeats < 10:
        raise ValueError("spec requires at least ten repeats per shape/block")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline, base_intervals = _capture(baseline_rows, expected_repeats)
    candidate, new_intervals = _capture(candidate_rows, expected_repeats)
    groups = []
    if baseline and candidate:
        if set(baseline) != set(candidate) or set(base_intervals) != set(new_intervals):
            raise ValueError("baseline/candidate frozen inputs or paired blocks differ")
        intervals = sorted([(start, end, role, block) for role, source in
                            (("baseline", base_intervals), ("candidate", new_intervals))
                            for block, (start, end) in source.items()])
        expected_order = [(role, block) for block in sorted(base_intervals) for role in ("baseline", "candidate")]
        if [(role, block) for _, _, role, block in intervals] != expected_order or any(
                intervals[index][1] >= intervals[index + 1][0] for index in range(len(intervals) - 1)):
            raise ValueError("blocks must be non-overlapping alternating baseline/candidate pairs")
        for digest, context, output in sorted(baseline, key=lambda key: (key[1], key[2], key[0])):
            key = (digest, context, output)
            flatten = lambda source: [row for repeats in source[key].values() for row in repeats.values() if not _failed(row)]
            base, new = _stats(flatten(baseline)), _stats(flatten(candidate))
            ci_ttft, ci_decode = _bootstrap(baseline[key], candidate[key], bootstrap_samples, seed)
            groups.append({"input_sha256": digest, "prompt_tokens": context, "requested_output_tokens": output,
                "paired_blocks": len(baseline[key]), "baseline": base, "candidate": new,
                "paired_completed_repeats_by_block": {str(block): sum(not _failed(row) and not _failed(candidate[key][block][repeat])
                    for repeat, row in repeats.items()) for block, repeats in baseline[key].items()},
                "ttft_improvement_percent": _improvement(base["ttft_ms"]["median"], new["ttft_ms"]["median"]),
                "decode_improvement_percent": _improvement(base["aggregate_decode_ms_per_token"], new["aggregate_decode_ms_per_token"]),
                "ttft_improvement_ci95_percent": ci_ttft, "decode_improvement_ci95_percent": ci_decode,
                "ttft_p95_regression_percent": (new["ttft_ms"]["p95"] / base["ttft_ms"]["p95"] - 1) * 100
                    if base["requests"] and new["requests"] else None,
                "decode_p95_regression_percent": (new["decode_ms_per_token"]["p95"] / base["decode_ms_per_token"]["p95"] - 1) * 100
                    if base["requests"] and new["requests"] else None,
                "changed_greedy_output_pairs": sum(
                    row["output_sha256"] != candidate[key][block][repeat]["output_sha256"]
                    for block, repeats in baseline[key].items() for repeat, row in repeats.items()
                    if not _failed(row) and not _failed(candidate[key][block][repeat])
                    and "output_sha256" in row and "output_sha256" in candidate[key][block][repeat])})
    supported = bool(groups) and all(group["paired_blocks"] >= 3 and all(count >= 10 for count in group["paired_completed_repeats_by_block"].values()) for group in groups)
    gates = {"sample_support": _gate("pass" if supported else "missing", "requires three alternating AB pairs and >=10 repeats/shape/block")}
    targets = [group for group in groups if group["prompt_tokens"] in (target_contexts or [])]
    if not supported or not targets or set(target_contexts or []) - {group["prompt_tokens"] for group in groups}:
        gates["serving_improvement"] = _gate("missing", "supported designated target workload required")
    else:
        wins = [any(group[f"{metric}_improvement_percent"] >= 5 - 1e-9 and
                    group[f"{metric}_improvement_ci95_percent"][0] > 0 for metric in ("ttft", "decode")) for group in targets]
        gates["serving_improvement"] = _gate("pass" if all(wins) else "fail", ">=5% TTFT or aggregate decode gain with paired CI excluding zero for each designated target")
    gates["protected_p95"] = _gate("missing" if not supported else "pass" if all(
        group[f"{metric}_p95_regression_percent"] <= 5 + 1e-9 for group in groups for metric in ("ttft", "decode")) else "fail", "<=5% p95 regression across measured protected workloads")
    speculation = {"baseline": _metrics_delta(baseline_metrics, {row["label"] for row in baseline_rows if row.get("warmup") is False}),
                   "candidate": _metrics_delta(candidate_metrics, {row["label"] for row in candidate_rows if row.get("warmup") is False})}
    if all(value["status"] == "measured" for value in speculation.values()):
        change = (speculation["candidate"]["acceptance_rate"] - speculation["baseline"]["acceptance_rate"]) * 100
        speculation["acceptance_change_percentage_points"] = change
        gates["mtp_acceptance"] = _gate("pass" if change >= -2 - 1e-9 else "fail", "no unexplained acceptance reduction beyond two percentage points")
    else:
        gates["mtp_acceptance"] = _gate("missing", "valid before/after draft and accepted counter deltas required")
    quality = _quality(baseline_quality, candidate_quality)
    gates["quality"] = _gate(quality["status"], quality["reason"])
    gates["quality_oracle"] = _quality_oracle(baseline_quality, candidate_quality)
    gates["stream_completion"] = _completion_gate(baseline_rows + candidate_rows)
    evidence = evidence or {}
    reduction = evidence.get("target_kernel_median_reduction_percent")
    gates["target_kernel"] = _gate("missing" if reduction is None else "pass" if isinstance(reduction, (float, int)) and math.isfinite(reduction) and reduction >= 10 else "fail", ">=10% target-kernel median reduction; separate benchmark evidence required")
    for name in ("activation_verified", "runtime_identity_frozen", "numerical_graph_checks_passed", "memory_gate_passed", "application_replay_passed"):
        value = evidence.get(name)
        gates[name] = _gate("pass" if value is True else "fail" if value is False else "missing", "separate evidence required")
    explicit_errors = []
    for source in (baseline_rows, candidate_rows):
        measured = [row for row in source if row.get("warmup") is False]
        explicit_errors.append(sum(_failed(row) for row in measured) if measured and all("complete" in row and "error" in row for row in measured) else None)
    errors = [evidence.get(name, explicit_errors[index]) for index, name in enumerate(("baseline_errors", "candidate_errors"))]
    gates["errors"] = _gate("missing" if any(type(value) is not int or value < 0 for value in errors) else "pass" if errors[1] <= errors[0] else "fail", "no increased errors; completed-request JSONL alone does not count failed attempts")
    greedy_rows = [[row for row in source if row.get("warmup") is False and not _failed(row)]
                   for source in (baseline_rows, candidate_rows)]
    greedy_complete = all(rows and all("output_sha256" in row for row in rows) for rows in greedy_rows)
    gates["greedy_evidence"] = _gate("pass" if greedy_complete else "missing",
        "valid output SHA256 required for every completed latency row in both captures")
    changed = quality.get("changed_outputs") or any(group["changed_greedy_output_pairs"] for group in groups)
    gates["greedy_output_review"] = _gate("missing" if not greedy_complete or changed and evidence.get("greedy_changes_reviewed") is not True else "pass", "complete greedy evidence and inspection of changed outputs required")
    status = "fail" if any(gate["status"] == "fail" for gate in gates.values()) else "incomplete" if any(gate["status"] == "missing" for gate in gates.values()) else "pass"
    return {"status": status, "groups": groups, "gates": gates, "speculation": speculation, "quality": quality,
            "baseline_errors": explicit_errors[0], "candidate_errors": explicit_errors[1],
            "bootstrap": {"method": "hierarchical paired blocks then paired repeats", "samples": bootstrap_samples, "seed": seed, "confidence": 0.95},
            "scope": "development 254; client TTFT is not prefill latency; aggregate decode is not stream-chunk token timing",
            "baseline_attempted_requests": sum(len(repeats) for blocks in baseline.values() for repeats in blocks.values()),
            "candidate_attempted_requests": sum(len(repeats) for blocks in candidate.values() for repeats in blocks.values()),
            "baseline_latency_rows": sum(not _failed(row) for blocks in baseline.values() for repeats in blocks.values() for row in repeats.values()),
            "candidate_latency_rows": sum(not _failed(row) for blocks in candidate.values() for repeats in blocks.values() for row in repeats.values())}


def _jsonl(paths):
    rows = [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def _json_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metrics/evidence JSON must be an object")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", default=[])
    parser.add_argument("--baseline-metrics", nargs="+", default=[])
    parser.add_argument("--candidate-metrics", nargs="+", default=[])
    parser.add_argument("--baseline-quality", nargs="+", default=[])
    parser.add_argument("--candidate-quality", nargs="+", default=[])
    parser.add_argument("--target-context", type=int, action="append")
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evidence", type=Path, help="JSON with external kernel/activation/runtime/error/graph/memory/application evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(_jsonl(args.baseline), _jsonl(args.candidate), expected_repeats=args.expected_repeats,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed, target_contexts=args.target_context,
            baseline_metrics=[_json_object(path) for path in args.baseline_metrics],
            candidate_metrics=[_json_object(path) for path in args.candidate_metrics],
            baseline_quality=_jsonl(args.baseline_quality), candidate_quality=_jsonl(args.candidate_quality),
            evidence=_json_object(args.evidence) if args.evidence else None)
        exit_code = 0
    except (ValueError, OSError) as error:
        report, exit_code = {"status": "invalid", "validation_errors": [str(error)]}, 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
