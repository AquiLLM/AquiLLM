"""CPU-only descriptive comparison of archived bounded API screens.

Reads selected JSON/JSONL members directly; never extracts archives or emits
raw prompts, generated text, logs, environment, or credentials.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def memory_summary(samples):
    good = [row for row in samples if 'gpu' in row]
    parsed = [[float(x) for x in row['gpu'].split(',')] for row in good]
    assert all(len(values) == 4 and all(math.isfinite(x) for x in values) for values in parsed)
    return dict(samples=len(good), errors=len(samples)-len(good),
                peak_used_mib=max((x[0] for x in parsed), default=None),
                min_free_mib=min((x[1] for x in parsed), default=None),
                scope='5-second whole-device samples; not allocator peaks or per-process attribution')


def performance(row, memory):
    errors = []
    results = row['results']
    frozen = row['frozen_identity']
    if identity(frozen['requests']) != frozen['sha256']: errors.append('frozen_identity_hash')
    if len(results) != row['requests'] or len(results) != len(frozen['requests']): errors.append('request_count')
    for item, result in zip(frozen['requests'], results):
        if any(result.get(k) != v for k, v in item.items()): errors.append('request_identity')
        if (not result['complete'] or result['error'] is not None or not result['stream_done']
                or result['finish_reason'] != 'length'): errors.append('stream_completion')
        usage = result['usage']
        if (result['output_tokens'] != item['requested_output_tokens']
                or usage['completion_tokens'] != item['requested_output_tokens']
                or usage['prompt_tokens'] != item['prompt_tokens']): errors.append('token_budget')
        if sha(result['output_text'].encode()) != result['output_sha256']: errors.append('output_hash')
    tokens = sum(r['output_tokens'] for r in results)
    if tokens != row['total_output_tokens']: errors.append('total_output_tokens')
    if not math.isclose(tokens / row['seconds'], row['output_tokens_per_second']): errors.append('throughput_arithmetic')
    decode = sum(r['total_seconds'] - r['ttft_seconds'] for r in results) / sum(r['output_tokens'] - 1 for r in results)
    if not math.isclose(decode, row['aggregate_decode_seconds_per_token']): errors.append('decode_arithmetic')
    delta = {k: row['metrics_after']['values'][k] - row['metrics_before']['values'][k]
             for k in ('drafts', 'draft_tokens', 'accepted_tokens', 'preemptions')}
    if any(delta[k] != row['speculation'][k] for k in delta): errors.append('counter_delta')
    start = datetime.fromisoformat(row['started_at']).timestamp()
    timed_memory = [s for s in memory if start <= s['time'] <= start + row['seconds']]
    latencies = [r['total_seconds'] for r in results]
    gauges = row['instrumentation']
    return dict(repeat=row['repeat'], complete=row['complete'], error=row['error'],
        independently_checked_errors=sorted(set(errors)), reported_validation_errors=row['validation_errors'],
        requests=len(results), frozen_identity=frozen, seconds=row['seconds'],
        output_tokens=tokens, output_tokens_per_second=tokens / row['seconds'],
        median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in results),
        worst_of_four_ttft_seconds=max(r['ttft_seconds'] for r in results),
        median_request_seconds=statistics.median(latencies), mean_request_seconds=statistics.mean(latencies),
        worst_request_seconds=max(latencies), summed_request_decode_seconds_per_token=decode,
        speculation=row['speculation'], sampled_gauges={k: gauges.get(k) for k in
            ('valid', 'errors', 'sampling_interval_seconds', 'max_running', 'max_waiting', 'max_cache_usage')},
        timed_memory=memory_summary(timed_memory),
        per_request=[{k: r[k] for k in ('id', 'input_sha256', 'prompt_tokens', 'requested_output_tokens',
                      'output_tokens', 'output_sha256', 'ttft_seconds', 'total_seconds', 'finish_reason')}
                     for r in results])


def lifecycle(rows):
    return [dict(round=r['round'], complete=r['complete'], passed=r['passed'], errors=r['errors'],
        sampled_running=r['overlap_sample']['values']['running'] if r.get('overlap_sample') else None,
        cancelled={k:r['cancelled'].get(k) for k in
            ('passed','nonempty_chunks','stream_done','finish_reason','close_method','cancelled_at')},
        survivor={k:r['survivor'].get(k) for k in ('passed','finish_reason','usage','started_at','captured_at')},
        recovery_count=len(r['recovery']), recovery_passed=sum(x['passed'] for x in r['recovery']),
        idle_sample_count=len(r['drain']['idle_samples']), final_metrics=r['after']['values'],
        server_abort_confirmation=r['server_abort_confirmation'], physical_state_slot_reuse=r['physical_state_slot_reuse'])
        for r in rows]


def load_archive(path):
    members, hashes = {}, {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(('.json', '.jsonl')): continue
            if name in members: raise ValueError('Duplicate evidence basename')
            raw = archive.extractfile(member).read()
            hashes[name] = sha(raw)
            members[name] = [json.loads(line) for line in raw.decode().splitlines()] if name.endswith('.jsonl') else json.loads(raw)
    memory = [r for name, rows in members.items() if name.endswith('-memory.jsonl') for r in rows]
    arms = {}
    for name, rows in members.items():
        match = re.fullmatch(r'(.+-(?:seq[124]|baseline))-(short|mixed)-c([124])\.jsonl', name)
        if not match: continue
        label, workload, concurrency = match.groups()
        arm = arms.setdefault(label, {})
        arm[workload + '-c' + concurrency] = [performance(row, memory) for row in rows]
    gates = {}
    for label in arms:
        quality = members.get(label + '-quality.jsonl', [])
        quality_identity_errors = []
        for row in quality:
            frozen=row['frozen_identity']
            if (identity(frozen['requests']) != frozen['sha256'] or
                    {'id':row['id'],'input_sha256':row['input_sha256']} not in frozen['requests']):
                quality_identity_errors.append(row['id'])
        gates[label] = dict(quality_count=len(quality), quality_passed=sum(r['passed'] for r in quality),
            quality_complete=sum(r['complete'] and r['error'] is None for r in quality),
            quality_frozen_hashes=sorted({r['frozen_identity']['sha256'] for r in quality}),
            quality_identity_errors=quality_identity_errors,
            quality_groups=dict(Counter(r['id'].rsplit('-',1)[0] for r in quality)),
            lifecycle=lifecycle(members.get(label + '-lifecycle.jsonl', [])),
            runtime={k:members.get(label + '-diagnostics.json', {}).get(k) for k in
                     ('image','running','health','restarts')})
    restored = [{k:value.get(k) for k in ('image','healthy','unrelated_services_unchanged','source_commit')}
                for name,value in members.items() if name.endswith('-restored.json')]
    graph_settings = [{k:value.get(k) for k in ('image','started','explicit_compilation_config','completed_capture_counts','scope')}
                      for name,value in members.items() if name.endswith('-graph-settings.json')]
    return dict(archive=path.as_posix(), archive_sha256=sha(path.read_bytes()), member_sha256=hashes,
                arms=arms, gates=gates, whole_run_memory=memory_summary(memory), restored=restored,
                graph_settings=graph_settings)


def cell(rows):
    """All repeats contribute; do not discard rows or invent boot replication."""
    requests = [request for row in rows for request in row['per_request']]
    drafts = sum(row['speculation']['draft_tokens'] for row in rows)
    return dict(repeats=len(rows), per_repeat_tok_s=[row['output_tokens_per_second'] for row in rows],
        per_repeat_seconds=[row['seconds'] for row in rows],
        output_tokens_per_repeat=[row['output_tokens'] for row in rows],
        output_tokens_per_second=sum(row['output_tokens'] for row in rows)/sum(row['seconds'] for row in rows),
        median_ttft_seconds=statistics.median(r['ttft_seconds'] for r in requests),
        median_request_seconds=statistics.median(r['total_seconds'] for r in requests),
        acceptance_rate=sum(row['speculation']['accepted_tokens'] for row in rows)/drafts,
        summed_request_decode_seconds_per_token=sum(r['total_seconds']-r['ttft_seconds'] for r in requests)/sum(r['output_tokens']-1 for r in requests),
        per_request_median_seconds={identifier:statistics.median(r['total_seconds'] for r in requests if r['id']==identifier)
                                    for identifier in sorted({r['id'] for r in requests})})


def compare(baseline, candidate, *, baseline_labels=None, candidate_labels=None):
    pairs = []
    for baseline_label, baseline_arm in baseline['arms'].items():
        if baseline_labels is not None and baseline_label not in baseline_labels: continue
        for candidate_label, candidate_arm in candidate['arms'].items():
            if candidate_labels is not None and candidate_label not in candidate_labels: continue
            for key in sorted(set(baseline_arm) & set(candidate_arm)):
                before, after = baseline_arm[key], candidate_arm[key]
                b,c = cell(before),cell(after)
                before_hashes = {identifier:{r['output_sha256'] for row in before for r in row['per_request'] if r['id']==identifier}
                                 for identifier in b['per_request_median_seconds']}
                matches = sum(r['output_sha256'] in before_hashes.get(r['id'],set()) for row in after for r in row['per_request'])
                pairs.append(dict(baseline_label=baseline_label,candidate_label=candidate_label,
                    workload_concurrency=key, baseline_summary=b,candidate_summary=c,
                    equal_frozen_inputs=all(row['frozen_identity']==before[0]['frozen_identity'] for row in before+after),
                    equal_output_budget_per_repeat=set(b['output_tokens_per_repeat'])==set(c['output_tokens_per_repeat']),
                    candidate_outputs_matching_a_baseline_hash=matches, candidate_output_count=sum(r['requests'] for r in after),
                    throughput_ratio=c['output_tokens_per_second']/b['output_tokens_per_second'],
                    median_ttft_change_seconds=c['median_ttft_seconds']-b['median_ttft_seconds'],
                    median_request_latency_ratio=c['median_request_seconds']/b['median_request_seconds'],
                    summed_decode_metric_ratio=c['summed_request_decode_seconds_per_token']/b['summed_request_decode_seconds_per_token'],
                    mtp_acceptance_change_percentage_points=100*(c['acceptance_rate']-b['acceptance_rate']),
                    per_request_latency_ratios={k:c['per_request_median_seconds'][k]/b['per_request_median_seconds'][k]
                                               for k in sorted(set(b['per_request_median_seconds'])&set(c['per_request_median_seconds']))}))
    return pairs


def table(pairs):
    lines = [
        '| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        key=pair['workload_concurrency']; b,c=pair['baseline_summary'],pair['candidate_summary']
        lines.append(f"| {key} | {b['output_tokens_per_second']:.2f} | {c['output_tokens_per_second']:.2f} | {pair['throughput_ratio']:.3f}x | {b['median_ttft_seconds']:.3f} / {c['median_ttft_seconds']:.3f} | {b['median_request_seconds']:.3f} / {c['median_request_seconds']:.3f} | {pair['mtp_acceptance_change_percentage_points']:+.2f} |")
    return lines


def markdown(report, command):
    lines = ['# Paired API throughput screen', '',
        'Descriptive fixed-work observations only. The initial comparison changes the registered plugin image and sequence limit together; it does not isolate a kernel speedup or qualify promotion.', '',
        'Evidence validation: **'+('passed' if report['valid_evidence'] else 'FAILED; inspect validation_errors in JSON')+'**.', '',
        'Throughput combines all repeat tokens divided by all repeat wall time; latency medians pool requests. Raw per-repeat values remain in JSON. Repeats within one boot are not independent deployment replications.', '']
    lines += table(report['comparisons'])
    lines += ['', '## Quality, lifecycle and resources', '',
              '| Archive / arm | Quality complete and passed | Lifecycle rounds passed | Recovery passed | Sampled whole-run used peak MiB / free minimum MiB |',
              '|---|---:|---:|---:|---:|']
    evidence=[('baseline',report['baseline']),('candidate',report['candidate'])]
    if 'graph_screen' in report: evidence.append(('graphs',report['graph_screen']['evidence']))
    for role,archive in evidence:
        memory=archive['whole_run_memory']
        for label,gates in archive['gates'].items():
            rounds=gates['lifecycle']
            quality=f"{gates['quality_passed']}/{gates['quality_count']}" if gates['quality_count'] else 'not measured'
            cancellation=f"{sum(r['passed'] for r in rounds)}/{len(rounds)}" if rounds else 'not measured'
            recovery=f"{sum(r['recovery_passed'] for r in rounds)}/{sum(r['recovery_count'] for r in rounds)}" if rounds else 'not measured'
            lines.append(f"| {role}: {label} | {quality} | {cancellation} | {recovery} | {memory['peak_used_mib']:.0f} / {memory['min_free_mib']:.0f} |")
    lines += ['', 'Quality includes 32 short strict/tool/multiturn/reasoning cases and six long-context cases where present. Recorded oracle passes are summarized; the archive comparison independently rechecks performance hashes, counts, timing arithmetic and speculation-counter deltas, rather than re-running the model or quality oracle.', '',
              'Lifecycle passes establish sampled overlap, socket disconnect, a correct surviving stream, idle metrics and six fresh recovery oracles. Server abort and physical state-slot reuse remain unobserved. No equivalent concurrent lifecycle run exists in the seq1 baseline archive.', '',
              'Memory is coarse whole-device sampling, including boots/rollback; timed-run samples are separated in JSON. Restoration records and per-arm health/restarts are also retained.', '',
              '## Mixed request latency tradeoffs', '',
              '| Clients / request | Prompt tokens | Baseline total seconds | Candidate total seconds | Ratio |',
              '|---|---:|---:|---:|---:|']
    for pair in report['comparisons']:
        if not pair['workload_concurrency'].startswith('mixed-'): continue
        b,c=pair['baseline_summary'],pair['candidate_summary']
        first=report['baseline']['arms'][pair['baseline_label']][pair['workload_concurrency']][0]['per_request']
        prompt={r['id']:r['prompt_tokens'] for r in first}
        for identifier,ratio in pair['per_request_latency_ratios'].items():
            lines.append(f"| {pair['workload_concurrency']} / {identifier} | {prompt[identifier]} | {b['per_request_median_seconds'][identifier]:.3f} | {c['per_request_median_seconds'][identifier]:.3f} | {ratio:.2f}x |")
    if 'graph_screen' in report:
        lines += ['', '## Fresh baseline C1 versus original baseline', '',
                  'This quantifies boot/history variation under the restored baseline; it is not an optimization arm.', ''] + table(report['graph_screen']['fresh_baseline_vs_original'])
        lines += ['', '## Explicit graph sizes versus fresh baseline C1', ''] + table(report['graph_screen']['fresh_baseline_comparisons'])
        lines += ['', 'Configured capture sizes and startup counts are retained in JSON when captured. They establish available graphs, not per-request replay selection.', '']
        lines += ['', '## Explicit graph sizes versus earlier candidate', '',
                  'Same plugin image, changed capture policy and later boot/history. Output and MTP variation still limit attribution.', ''] + table(report['graph_screen']['prior_candidate_comparisons'])
        lines += ['', '## Explicit graph sizes versus original baseline', '',
                  'Concurrent comparisons use the earlier baseline archive; a fresh concurrent baseline was not measured.', ''] + table(report['graph_screen']['original_baseline_comparisons'])
    lines += ['', 'Inputs and returned token budgets are checked separately from generated-text identity. See the JSON for every output hash, raw-evidence hash, request latency and validation result.', '',
              '## Evidence limits', ''] + ['- '+x for x in report['limits']]
    lines += ['', '## Reproduce', '', '```powershell', command, '```', '']
    return '\n'.join(lines)


def validation_errors(evidence):
    errors=[]
    if not evidence['arms']: errors.append('missing_performance_arms')
    for label,arm in evidence['arms'].items():
        for key,rows in arm.items():
            if not rows: errors.append('missing_performance_repeats')
            for row in rows:
                errors.extend(row['independently_checked_errors'])
                if row['complete'] is not True or row['error'] is not None or row['reported_validation_errors']:
                    errors.append('reported_performance_invalid')
                gauges=row['sampled_gauges']
                if gauges['valid'] is not True or gauges['errors']: errors.append('reported_instrumentation_invalid')
        gates=evidence['gates'][label]
        if gates['quality_identity_errors']: errors.append('quality_identity_invalid')
        if gates['quality_count'] and (gates['quality_count']!=38 or
                gates['quality_complete']!=38 or gates['quality_passed']!=38):
            errors.append('quality_gate_failed')
        for row in gates['lifecycle']:
            if (row['complete'] is not True or row['passed'] is not True or row['errors'] or
                    row['recovery_count']!=6 or row['recovery_passed']!=6): errors.append('lifecycle_gate_failed')
        runtime=gates['runtime']
        if runtime['image'] is not None and (runtime['running'] is not True or runtime['health']!='healthy' or runtime['restarts']!=0):
            errors.append('runtime_gate_failed')
    if not evidence['restored']: errors.append('missing_final_restoration')
    for row in evidence['restored']:
        if row['healthy'] is not True or row['unrelated_services_unchanged'] is not True:
            errors.append('restoration_gate_failed')
    return errors


def comparison_validation_errors(result):
    """Require this screen's full comparison matrix and matched workloads."""
    full={f'{workload}-c{clients}' for workload in ('short','mixed') for clients in (1,2,4)}
    c1={'short-c1','mixed-c1'}
    groups={'initial':(result['comparisons'],full)}
    if 'graph_screen' in result:
        graph=result['graph_screen']
        for name,expected in (('fresh_baseline_vs_original',c1),('fresh_baseline_comparisons',c1),
                ('prior_candidate_comparisons',full),('original_baseline_comparisons',full)):
            groups[name]=(graph.get(name,[]),expected)
    errors=[]
    for name,(pairs,expected) in groups.items():
        if len(pairs)!=len(expected) or {p['workload_concurrency'] for p in pairs}!=expected:
            errors.append(f'{name}:missing_or_duplicate_comparison_cells')
        for pair in pairs:
            if pair['equal_frozen_inputs'] is not True:
                errors.append(f"{name}:{pair['workload_concurrency']}:frozen_inputs_differ")
            if pair['equal_output_budget_per_repeat'] is not True:
                errors.append(f"{name}:{pair['workload_concurrency']}:output_budgets_differ")
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--graph-screen',type=Path)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args()
    baseline,candidate=load_archive(args.baseline),load_archive(args.candidate)
    result=dict(baseline=baseline,candidate=candidate,comparisons=compare(baseline,candidate),
        qualification='descriptive_screen_only',limits=[
        'Only the recorded repeats contribute; four requests per repeat are not independent boot replicates. No confidence interval or promotion claim.',
        'Fixed arm/workload order, warmup/history, separate boots, output divergence and MTP acceptance confound causal attribution.',
        'Aggregate output tokens/workload wall second includes prompt processing and client queueing; it is not pure decode kernel throughput.',
        'Summed request decode time/token counts overlapping requests separately and includes stalls after first output; it is not reciprocal aggregate throughput or measured inter-token latency.',
        'Each repeat reports nearest-rank p95 with four requests, which equals the maximum; it is not a stable tail estimate.',
        'Metrics sampled at 0.2 seconds show request overlap and cache utilization, not simultaneous token execution or exact peaks. Counters require isolated traffic.',
        '5-second nvidia-smi samples can miss peaks and cover the entire device, including startup/rollback outside timed windows.',
        'Strict synthetic oracles and bounded disconnect/recovery do not establish general quality, application replay, scheduler abort or physical state-slot reuse.',
        'The initial archives contain seq1 baseline versus seq4 candidate; no seq2 server-limit arm is inferred.'])
    if args.graph_screen:
        graphs=load_archive(args.graph_screen)
        baseline_labels=[label for label in graphs['arms'] if label.endswith('-baseline')]
        candidate_labels=[label for label in graphs['arms'] if not label.endswith('-baseline')]
        result['graph_screen']=dict(evidence=graphs,
            fresh_baseline_vs_original=compare(baseline,graphs,candidate_labels=baseline_labels),
            fresh_baseline_comparisons=compare(graphs,graphs,baseline_labels=baseline_labels,candidate_labels=candidate_labels),
            prior_candidate_comparisons=compare(candidate,graphs,candidate_labels=candidate_labels),
            original_baseline_comparisons=compare(baseline,graphs,candidate_labels=candidate_labels))
    script_relative=Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[5]).as_posix()
    command=f'rtk proxy python {script_relative} --baseline {args.baseline.as_posix()} --candidate {args.candidate.as_posix()} --output-prefix {args.output_prefix.as_posix()}'
    if args.graph_screen: command+=' --graph-screen '+args.graph_screen.as_posix()
    errors={role:validation_errors(arm) for role,arm in
            ([('baseline',baseline),('candidate',candidate),('graphs',graphs)] if args.graph_screen else [('baseline',baseline),('candidate',candidate)])}
    errors['comparisons']=comparison_validation_errors(result)
    result['validation_errors']=errors
    result['valid_evidence']=not any(errors.values())
    args.output_prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.output_prefix.with_suffix('.md').write_text(markdown(result,command),encoding='utf-8')
    comparison_counts={'initial':len(result['comparisons'])}
    if args.graph_screen:
        comparison_counts.update({k:len(v) for k,v in result['graph_screen'].items() if k!='evidence'})
    print(json.dumps(dict(compared_cells=comparison_counts,validation_errors=errors,valid_evidence=result['valid_evidence'])))
    return 0 if result['valid_evidence'] else 1


if __name__=='__main__': raise SystemExit(main())
