"""Read-only exploratory allocator analysis; three paired boots are the units.

No network, Docker, subprocesses, model calls, or source changes. Output contains
statistics and digests, never captured model text or process environment values.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import percentile, _metrics_delta


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid(row):
    return (row.get("complete") is True and row.get("error") is None and row.get("stream_done") is True
            and row.get("finish_reason") == "length" and type(row.get("output_tokens")) is int
            and row["output_tokens"] == row.get("requested_output_tokens")
            and isinstance(row.get("usage"), dict) and row["usage"].get("completion_tokens") == row["output_tokens"]
            and all(type(row.get(name)) in (float, int) and math.isfinite(row[name]) and row[name] >= 0
                    for name in ("ttft_seconds", "total_seconds", "aggregate_decode_seconds_per_token"))
            and row["total_seconds"] > row["ttft_seconds"])


def shape(row):
    return row.get("input_sha256"), row.get("prompt_tokens"), row.get("requested_output_tokens")


def summary(rows):
    ttft = [row["ttft_seconds"] * 1000 for row in rows]
    total = [row["total_seconds"] * 1000 for row in rows]
    decode = [row["aggregate_decode_seconds_per_token"] * 1000 for row in rows]
    wall = sum(row["total_seconds"] for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    return dict(requests=len(rows), output_tokens=tokens, total_wall_request_seconds=wall,
        ttft_ms=dict(median=statistics.median(ttft), p95=percentile(ttft, .95)),
        total_ms=dict(median=statistics.median(total), p95=percentile(total, .95)),
        decode_ms_per_token=dict(median=statistics.median(decode), p95=percentile(decode, .95),
            aggregate=sum(row["total_seconds"] - row["ttft_seconds"] for row in rows) * 1000 /
                      sum(row["output_tokens"] - 1 for row in rows)),
        output_tokens_per_second=tokens / wall)


def uncertainty(ratios, *, latency=False):
    """Ratios >1 favor mimalloc; exactly one observation per paired boot."""
    if len(ratios) != 3 or any(value <= 0 or not math.isfinite(value) for value in ratios):
        return dict(status="missing", reason="exactly three finite paired boot effects required")
    logs = [math.log(value) for value in ratios]
    mean = statistics.mean(logs)
    margin = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
    convert = (lambda value: (1 - math.exp(-value)) * 100) if latency else (lambda value: (math.exp(value) - 1) * 100)
    bootstrap = [convert(statistics.mean(sample)) for sample in itertools.product(logs, repeat=3)]
    observed = abs(sum(logs))
    sign_flip = [abs(sum(sign * value for sign, value in zip(signs, logs)))
                     for signs in itertools.product((-1, 1), repeat=3)]
    p = sum(value >= observed - 1e-12 for value in sign_flip) / len(sign_flip)
    return dict(status="exploratory", paired_boots=3, improvement_percent=convert(mean),
        effect_definition="latency reduction percent" if latency else "throughput increase percent",
        block_effects_percent=[convert(value) for value in logs],
        block_t_95_percent=[convert(mean - margin), convert(mean + margin)],
        exact_block_bootstrap_95_percent=[percentile(bootstrap, q) for q in (.025, .975)],
        paired_sign_flip_sensitivity_two_sided_p=p, minimum_possible_two_sided_p=.25,
        sign_flip_assumptions="paired-effect sign exchangeability/symmetry; this is sensitivity analysis, not randomized-treatment evidence",
        assumptions="paired-block t interval assumes independent approximately normal log boot effects; df=2",
        limitation="three boots are exploratory; bootstrap has only 27 resamples; fixed system-then-mimalloc order leaves time/order drift unresolved")


def process_memory(value, *, expected_allocator=None):
    """Extract only named numeric memory fields and mapped-library counts."""
    measurements = defaultdict(list)
    mapped_nodes = set()
    records = []
    def collect_kb(fields):
        for name in ("VmRSS", "VmHWM", "Rss", "Pss", "RssAnon", "Private_Dirty"):
            amount = fields.get(name)
            match = re.fullmatch(r"(\d+)\s*kB", amount.strip()) if isinstance(amount, str) else None
            if match:
                measurements[name.lower() + "_kb"].append(int(match[1]))
    def walk(node):
        if isinstance(node, dict):
            libraries = node.get("libraries")
            library_mapped = isinstance(libraries, list) and any(
                isinstance(library, str) and re.search(r"libmimalloc[^\s]*\.so", library) for library in libraries)
            if library_mapped: mapped_nodes.add(id(node))
            if node.get("role") in ("api", "engine"):
                records.append((node, bool(library_mapped)))
            for name, item in node.items():
                if name.lower() in ("rss_kb", "rss_mib", "vmrss_kb", "vmhwm_kb", "pss_kb", "gpu_memory_mib") and type(item) in (int, float):
                    measurements[name.lower()].append(item)
                if name.lower() in ("maps", "mappings") and isinstance(item, str) and re.search(r"libmimalloc[^\s]*\.so", item):
                    mapped_nodes.add(id(node))
                if name.lower() in ("status", "memory") and isinstance(item, dict):
                    collect_kb(item)
                if name.lower() in ("status", "smaps_rollup") and isinstance(item, str):
                    for field, amount in re.findall(r"(?m)^(VmRSS|VmHWM|Rss|Pss|RssAnon):\s*(\d+)\s*kB", item):
                        measurements[field.lower() + "_kb"].append(int(amount))
                walk(item)
        elif isinstance(node, list):
            for item in node: walk(item)
    walk(value)
    reasons = []
    roles = {record.get("role") for record, _ in records}
    unreliable_engines = 0
    if expected_allocator in ("system", "mimalloc"):
        if roles != {"api", "engine"}: reasons.append("missing_api_or_engine_role")
        for record, library_mapped in records:
            role = record["role"]
            if record.get("configured_allocator") != expected_allocator:
                reasons.append(role + "_configured_allocator_mismatch")
            if not isinstance(record.get("libraries"), list) or library_mapped != (expected_allocator == "mimalloc"):
                reasons.append(role + "_library_mapping_mismatch")
            reliable = record.get("process_environment_reliable")
            if role == "api" and reliable is not True:
                reasons.append("api_environment_unreliable")
            if role == "api" or reliable is True:
                if record.get("observed_env_allocator") != expected_allocator or record.get("observed_env_pythonmalloc") != "default":
                    reasons.append(role + "_environment_mismatch")
            elif reliable is False:
                unreliable_engines += 1
            else:
                reasons.append("engine_environment_reliability_missing")
    activation = dict(status=("failed" if reasons else "verified_for_captured_roles") if expected_allocator else "missing_expected_allocator",
        expected_allocator=expected_allocator, captured_role_counts={role: sum(record.get("role") == role for record, _ in records) for role in ("api", "engine")},
        failures=sorted(set(reasons)), unreliable_engine_environment_records=unreliable_engines,
        note="engine /proc environment marked unreliable is inconclusive; API environment and both roles' library mappings remain strict")
    return dict(status="captured" if measurements else "missing_known_memory_fields",
                memory_fields={name: dict(records=len(items), sum=sum(items), maximum=max(items)) for name, items in measurements.items()},
                processes_with_mapped_mimalloc=len(mapped_nodes), activation=activation,
                limitation="before/after endpoints are not a memory soak or leak test; mapping count alone does not prove every serving child is covered")


def build(directory, expected_repeats=10):
    groups = defaultdict(dict)
    mixed = defaultdict(list)
    errors, pairing, metrics, processes = [], [], {}, {}
    outputs = defaultdict(dict)
    shape_sets = {}
    for role in ("system", "mimalloc"):
        for block in range(1, 4):
            path = directory / f"h100-allocator-{role}-block{block}.jsonl"
            if not path.exists():
                errors.append(dict(role=role, block=block, reason="missing_serving_file"))
                continue
            all_rows = read_jsonl(path)
            measured = [row for row in all_rows if row.get("warmup") is not True and type(row.get("repeat")) is int and row["repeat"] >= 0]
            by_shape = defaultdict(list)
            labels = {row.get("label") for row in all_rows}
            for row in measured:
                key = shape(row)
                if (re.fullmatch(r"[0-9a-f]{64}", key[0] or "") is None
                        or type(key[1]) is not int or type(key[2]) is not int or key[1] < 1 or key[2] < 2):
                    errors.append(dict(role=role, block=block, reason="invalid_frozen_shape_identity"))
                    continue
                by_shape[key].append(row)
                output_key = (block, key, row["repeat"])
                if role in outputs[output_key]: errors.append(dict(role=role, block=block, reason="duplicate_repeat"))
                outputs[output_key][role] = row.get("output_sha256")
                if row.get("output_text") is not None and hashlib.sha256(row["output_text"].encode()).hexdigest() != row.get("output_sha256"):
                    errors.append(dict(role=role, block=block, reason="invalid_output_digest"))
            for key, rows in by_shape.items():
                repeats = [row["repeat"] for row in rows]
                complete = [row for row in rows if valid(row)]
                if sorted(repeats) != list(range(expected_repeats)):
                    errors.append(dict(role=role, block=block, prompt_tokens=key[1], reason="missing_or_duplicate_repeats"))
                if len(complete) != len(rows):
                    errors.append(dict(role=role, block=block, prompt_tokens=key[1], reason="incomplete_or_invalid_requests", failed=len(rows)-len(complete)))
                if complete:
                    groups[key][role, block] = summary(complete)
                    mixed[role, block].extend(complete)
            shape_sets[role, block] = set(by_shape)
            candidates = list(directory.glob(path.stem + "-*-metrics.json"))
            if len(candidates) == 1:
                snapshot = json.loads(candidates[0].read_text(encoding="utf-8"))
                metrics[f"{role}-block{block}"] = _metrics_delta([snapshot], labels)
                metrics[f"{role}-block{block}"]["per_shape"] = [dict(
                    prompt_tokens=item.get("prompt_tokens"), input_sha256=item.get("input_sha256"),
                    requested_output_tokens=item.get("requested_output_tokens"),
                    counters=_metrics_delta([dict(label=snapshot.get("label"), before=item.get("before"), after=item.get("after"))], labels))
                    for item in snapshot.get("shapes", [])]
            else:
                metrics[f"{role}-block{block}"] = dict(status="missing", reason="exactly one per-block metrics file required")
    for (block, key, repeat), arms in outputs.items():
        if set(arms) != {"system", "mimalloc"}:
            pairing.append(dict(block=block, prompt_tokens=key[1], repeat=repeat, reason="unpaired_frozen_input_shape"))
        elif any(value is None or re.fullmatch(r"[0-9a-f]{64}", value or "") is None for value in arms.values()):
            pairing.append(dict(block=block, prompt_tokens=key[1], repeat=repeat, reason="missing_output_digest"))
        elif arms["system"] != arms["mimalloc"]:
            pairing.append(dict(block=block, prompt_tokens=key[1], repeat=repeat, reason="output_digest_difference"))
    if shape_sets and any(values != next(iter(shape_sets.values())) for values in shape_sets.values()):
        errors.append(dict(reason="frozen_shapes_differ_between_boots"))
    shapes = []
    for key, blocks in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], str(item[0][0]))):
        row = dict(input_sha256=key[0], prompt_tokens=key[1], requested_output_tokens=key[2],
                   blocks={f"{role}-block{block}": value for (role, block), value in blocks.items()})
        effects = {}
        for name, metric, field in (("ttft_median", "ttft_ms", "median"), ("total_median", "total_ms", "median"),
                                   ("decode_p95", "decode_ms_per_token", "p95"), ("total_p95", "total_ms", "p95")):
            ratios = [blocks["system", block][metric][field] / blocks["mimalloc", block][metric][field]
                      for block in range(1, 4) if ("system", block) in blocks and ("mimalloc", block) in blocks]
            effects[name] = uncertainty(ratios, latency=True)
        effects["output_throughput"] = uncertainty([
            blocks["mimalloc", block]["output_tokens_per_second"] / blocks["system", block]["output_tokens_per_second"]
            for block in range(1, 4) if ("system", block) in blocks and ("mimalloc", block) in blocks])
        row["paired_boot_effects"] = effects
        shapes.append(row)
    mixed_summaries = {f"{role}-block{block}": summary(rows) for (role, block), rows in mixed.items()}
    mixed_effect = uncertainty([mixed_summaries[f"mimalloc-block{block}"]["output_tokens_per_second"] /
                               mixed_summaries[f"system-block{block}"]["output_tokens_per_second"]
                               for block in range(1, 4) if f"system-block{block}" in mixed_summaries and f"mimalloc-block{block}" in mixed_summaries])
    mixed_latency = uncertainty([mixed_summaries[f"system-block{block}"]["total_wall_request_seconds"] /
                                 mixed_summaries[f"mimalloc-block{block}"]["total_wall_request_seconds"]
                                 for block in range(1, 4) if f"system-block{block}" in mixed_summaries and f"mimalloc-block{block}" in mixed_summaries], latency=True)
    for path in directory.glob("h100-allocator-*.json"):
        match = re.search(r"h100-allocator-(system|mimalloc)-block([1-3])", path.name)
        phase = re.search(r"(?:^|[-_])(before|after)(?:[-_.]|$)", path.name)
        if match and phase and not path.name.endswith("-metrics.json"):
            identity = f"{match[1]}-block{match[2]}-{phase[1]}"
            if identity in processes:
                errors.append(dict(reason="duplicate_process_snapshot", identity=identity))
            processes[identity] = process_memory(json.loads(path.read_text(encoding="utf-8")), expected_allocator=match[1])
    expected_processes = {f"{role}-block{block}-{phase}" for role in ("system", "mimalloc") for block in range(1, 4) for phase in ("before", "after")}
    missing_processes = sorted(expected_processes - processes.keys())
    statistical_screen = (mixed_effect.get("status") == "exploratory" and mixed_effect["improvement_percent"] >= 5
                          and mixed_effect["block_t_95_percent"][0] > 0)
    return dict(status="exploratory_analysis_only", treatment_units="paired boot blocks; warmups excluded from serving latency",
        allocator_order="fixed system then mimalloc in each pair; neither randomized nor counterbalanced; time/order drift unresolved",
        expected_repeats=expected_repeats, serving_errors=errors, prompt_output_pairing_issues=pairing,
        shapes=shapes, mixed_per_block=mixed_summaries, mixed_boot_uncertainty=mixed_effect,
        mixed_wall_latency_uncertainty=mixed_latency,
        speculation=metrics, process_memory=processes, missing_process_snapshots=missing_processes,
        protocol_screen=dict(gain_at_least_5_and_block_t_interval_excludes_zero=statistical_screen,
            wall_latency_reduction_at_least_5_and_block_t_interval_excludes_zero=(
                mixed_latency.get("status") == "exploratory" and mixed_latency["improvement_percent"] >= 5
                and mixed_latency["block_t_95_percent"][0] > 0),
            request_evidence_complete=not errors and not pairing,
            qualification="pending independent review of p95/errors/output/quality/activation/runtime/memory, fixed-order drift, and application replay",
            limitation="three boot pairs cannot support a distribution-free 95% improvement claim; exact two-sided minimum p=.25"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="local copied /tmp captures; read only")
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_repeats < 1: parser.error("expected repeats must be positive")
    result = build(args.directory, args.expected_repeats)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output: args.output.write_text(encoded, encoding="utf-8")
    else: print(encoded, end="")


if __name__ == "__main__": main()
