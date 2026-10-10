# Serving comparison report

`scripts/h100_performance/report.py` compares frozen direct-serving captures
without contacting a server. The accompanying measurements concern development
254 only. `reproduce_report.py` rebuilds the saved comparison from its captured
inputs and explicit external evidence.

## Input and calculation contract

Pass all baseline JSONL files to `--baseline` and all candidate files to
`--candidate`. Labels must end with `-blockN`; matching N is one AB pair.
The timestamps must show nonoverlapping baseline/candidate alternation.
Grouping uses the frozen input SHA256, prompt token count and requested output
size. Every group must have the expected repeat IDs in every block, with no
duplicates. The default is ten repeats; smaller sets do not meet the spec.
Warmups are excluded before validation and statistics.

Explicit incomplete/error attempts remain counted and are excluded from all
latency statistics and paired bootstrap samples. Old discovery captures without
completion/error qualification can supply descriptive timing statistics, but
the stream-completion and error gates remain missing. Qualified records require
DONE, terminal `length`, matching completion usage, `complete=true`, and no error.
Missing repeats, mismatched inputs, bad timing, wrong JSON shapes, nonfinite
values, unexpected output sizes and nonalternating blocks produce an invalid
report rather than a comparison. Invalid CLI input returns exit code 2; valid
reports return 0, with pass/fail/incomplete expressed in the report itself.

Reports include median and interpolated p95 TTFT, median/p95 request-level
aggregate decode time, weighted aggregate decode milliseconds/token, and output
tokens divided by total request duration. Output throughput includes TTFT.
Decode uses total-minus-TTFT divided by output-minus-one; SSE chunk counts are
not interpreted as token timestamps. Client TTFT does not measure prefill alone.

The deterministic bootstrap resamples matched AB block pairs, then matched
repeat IDs within each chosen pair. Its default seed is 17 and sample count
2000. Confidence intervals are percentile 95% intervals for relative median
TTFT improvement and weighted aggregate decode improvement. Fewer than three
AB pairs, or fewer than ten completed paired requests in any block/shape, leaves
the statistical support gate incomplete.

## Evidence and thresholds

Design thresholds are enforced separately: target kernel median reduction at
least 10%; designated serving TTFT or aggregate decode gain at least 5% with
paired CI excluding zero; at most 5% p95 regression across the measured protected
shapes; no increased errors; at most two percentage points acceptance loss;
and preservation of all passing frozen quality checks. `--target-context` names
the designated workload. All measured shapes are protected.

Metrics options accept the serving harness's before/after JSON snapshots.
Snapshot labels must cover every serving block. The parser recognizes actual
`_total` counters, ignores `_created` timestamps and does not add per-position
accepted counters to total accepted tokens. It computes deltas before combining
intervals. Counter resets, changed/missing series or zero draft denominators are
missing evidence. These captured intervals include warmups, which is explicit
in the report.

Quality options accept JSONL for the fixed 32 number/tool/multiturn/reasoning
cases. Lost baseline passes fail the quality gate. Strict qualification requires
the harness's `exact-v1` oracle, completion/error fields and matching frozen
input/case hashes. Changed greedy outputs require separate inspection.

`--evidence` optionally supplies separate benchmark/provenance results:
`target_kernel_median_reduction_percent`, `activation_verified`,
`runtime_identity_frozen`, `numerical_graph_checks_passed`, `memory_gate_passed`,
`application_replay_passed`, and `greedy_changes_reviewed`. Explicit
`baseline_errors`/`candidate_errors` can supplement captured attempt counts.
Absent evidence stays missing. The reporter does not infer activation, numerical
correctness, memory safety or application latency from serving speed.

## Additional diagnostics and verification

Later captures also include a `shapes` array with per-prompt-length before/after
MTP counters. These scrapes occur outside timed requests and include warmups.
The existing blockwide acceptance calculation and promotion thresholds remain
unchanged. Block1 predates this additional diagnostic field.

The saved latency rows deduplicate generated text into `serving/outputs.json`,
keyed by its SHA256. Timing, completion, error, usage, and output hashes remain
in every row. `serving/capture-provenance.json` records original and transformed
file hashes and any output hashes whose text was never captured.

See [verification.json](verification.json) for the tested source revisions,
CPU/GPU counts, sanitizer results, and scope limitations. A valid report can
still be `fail` or `incomplete`; successful script execution is not a rollout
approval.
