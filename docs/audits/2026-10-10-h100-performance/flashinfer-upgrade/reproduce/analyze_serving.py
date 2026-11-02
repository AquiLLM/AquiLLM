"""Descriptive screening comparison; never declares statistical qualification."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

repo = next(parent for parent in Path(__file__).resolve().parents
            if (parent/'scripts/h100_performance/report.py').is_file())
sys.path.insert(0, str(repo/'scripts/h100_performance'))
from report import _metrics_delta

out = Path(sys.argv[1])
prefix = sys.argv[2]
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
result = {'prefix':prefix,'kind':'screening_not_statistical_qualification','arms':{}}
coverage = {}
for arm in ('baseline','candidate'):
    summary = {}
    for stage in ('strict','long'):
        captures = [r for p in out.glob(f'{prefix}-{arm}-block*-{stage}.jsonl') for r in rows(p)]
        summary[stage] = {'requests':len(captures),'passed':sum(r.get('passed') is True for r in captures),
                          'failed_cases':[r['id'] for r in captures if not r.get('passed')]}
    for stage in ('latency','decode'):
        groups = defaultdict(list)
        seen = set()
        labels = set()
        for p in out.glob(f'{prefix}-{arm}-block*-{stage}.jsonl'):
            for row in rows(p):
                if not row.get('warmup'):
                    if not row.get('complete') or row.get('error'):raise ValueError('Incomplete measurement')
                    key = (row['label'].split('-block')[-1],row['prompt_tokens'],row['requested_output_tokens'],row['repeat'])
                    if key in seen:raise ValueError('Duplicate measurement')
                    seen.add(key)
                    labels.add(row['label'])
                    groups[(row['prompt_tokens'],row['requested_output_tokens'])].append(row)
        coverage[(arm,stage)] = seen
        summary[stage] = {str(k):{'requests':len(v),
            'median_ttft_s':statistics.median(r['ttft_seconds'] for r in v),
            'median_total_s':statistics.median(r['total_seconds'] for r in v),
            'aggregate_decode_ms_per_token':1000*sum(r['total_seconds']-r['ttft_seconds'] for r in v)/sum(r['output_tokens']-1 for r in v),
            'output_tokens_per_second':sum(r['output_tokens'] for r in v)/sum(r['total_seconds'] for r in v),
            'output_hashes':sorted(set(r['output_sha256'] for r in v)),
            'input_hashes':sorted(set(r['input_sha256'] for r in v))} for k,v in groups.items()}
        all_rows = [r for values in groups.values() for r in values]
        summary[stage+'_total'] = {'requests':len(all_rows),
            'serial_output_tokens_per_second':sum(r['output_tokens'] for r in all_rows)/sum(r['total_seconds'] for r in all_rows)}
        snapshots = [json.loads(p.read_text()) for p in out.glob(f'{prefix}-{arm}-block*-{stage}-*-metrics.json')]
        summary[stage+'_mtp'] = _metrics_delta(snapshots, labels)
    load = [r for p in sorted(out.glob(f'{prefix}-{arm}-block*-load.jsonl')) for r in rows(p)]
    summary['load'] = [{'block':r['label'].split('-block')[-1],
        **{k:r[k] for k in ('concurrency','requests','seconds','output_tokens_per_second','median_ttft_seconds','input_sha256')}}
        for r in load]
    summary['load_mtp'] = [{ 'concurrency':r['concurrency'],
        **_metrics_delta([r], {r['label']}), 'interval_includes_warmups':False}
        for r in load]
    result['arms'][arm] = summary
comparison = {}
base,new = (result['arms'][arm] for arm in ('baseline','candidate'))
for stage in ('latency','decode'):
    if not coverage[('baseline',stage)] or coverage[('baseline',stage)] != coverage[('candidate',stage)]:
        raise ValueError('Measurement shape/block/repeat coverage mismatch')
    comparison[stage] = {}
    for shape, b in base[stage].items():
        n = new[stage][shape]
        if b['input_hashes'] != n['input_hashes']:raise ValueError('Input mismatch')
        comparison[stage][shape] = {
            'ttft_reduction_percent':100*(1-n['median_ttft_s']/b['median_ttft_s']),
            'total_latency_reduction_percent':100*(1-n['median_total_s']/b['median_total_s']),
            'decode_latency_reduction_percent':100*(1-n['aggregate_decode_ms_per_token']/b['aggregate_decode_ms_per_token']),
            'serial_throughput_increase_percent':100*(n['output_tokens_per_second']/b['output_tokens_per_second']-1),
            'same_output_hashes':b['output_hashes']==n['output_hashes']}
    comparison[stage+'_serial_throughput_increase_percent'] = 100*(new[stage+'_total']['serial_output_tokens_per_second']/base[stage+'_total']['serial_output_tokens_per_second']-1)
base_load = {(r['block'],r['concurrency']):r for r in base['load']}
new_load = {(r['block'],r['concurrency']):r for r in new['load']}
if (not base_load or set(base_load) != set(new_load) or
    len(base_load) != len(base['load']) or len(new_load) != len(new['load'])):
    raise ValueError('Load matrix mismatch')
comparison['load'] = []
for key,b in sorted(base_load.items()):
    n = new_load[key]
    if (b['concurrency'],b['requests'],b['input_sha256']) != (n['concurrency'],n['requests'],n['input_sha256']):raise ValueError('Load input mismatch')
    comparison['load'].append({'concurrency':b['concurrency'],
        'throughput_increase_percent':100*(n['output_tokens_per_second']/b['output_tokens_per_second']-1),
        'median_ttft_reduction_percent':100*(1-n['median_ttft_seconds']/b['median_ttft_seconds'])})
result['comparison'] = comparison
print(json.dumps(result,indent=2))
