"""Frozen concurrent serving screen; profiler runs must be measured separately.

Performance JSONL contains one summary plus raw requests/metrics per repeat.
Quality JSONL contains exactly 32 strict and six long-context oracle rows.
Polling proves only sampled overlap, not scheduler abort or state-slot reuse.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.request

from long_quality_bench import long_cases
from quality_bench import cases, evaluate, identity
from serve_bench import request, stream_request

SHAPES = {'short': [(512, 1024)], 'long': [(8192, 512)],
          'mixed': [(512, 256), (8192, 256), (32768, 256), (36864, 256)]}
SEED_TEXT = ('A synthetic performance note: the project has twelve numbered stages. '
             'Summarize the stages and explain their order. ')
POLL_INTERVAL = 0.2
METRIC_NAMES = {
    'vllm:num_requests_running': 'running', 'vllm:num_requests_waiting': 'waiting',
    'vllm:kv_cache_usage_perc': 'cache_usage', 'vllm:gpu_cache_usage_perc': 'cache_usage',
    'vllm:num_preemptions_total': 'preemptions',
    'vllm:spec_decode_num_drafts_total': 'drafts',
    'vllm:spec_decode_num_draft_tokens_total': 'draft_tokens',
    'vllm:spec_decode_num_accepted_tokens_total': 'accepted_tokens',
}
METRIC_LINE = re.compile(r'^([^\s{]+)(\{(?:[^"{}]|"(?:\\.|[^"\\])*")*\})?\s+(\S+)(?:\s+\S+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def captured_at():
    return datetime.now(timezone.utc).isoformat()


def freeze_workload(seed_tokens, model, workload, count):
    if (not seed_tokens or any(type(t) is not int or t < 0 for t in seed_tokens)
            or workload not in SHAPES or type(count) is not int or count < 1
            or count % len(SHAPES[workload])):
        raise ValueError('Invalid token seed or incomplete workload matrix')
    result = []
    for index in range(count):
        prompt, output = SHAPES[workload][index % len(SHAPES[workload])]
        ids = (seed_tokens * (prompt // len(seed_tokens) + 1))[:prompt]
        payload = dict(model=model, prompt=ids, max_tokens=output, temperature=0, seed=17,
                       ignore_eos=True, stream=True, stream_options={'include_usage': True})
        result.append(dict(id=f'request-{index}', prompt_tokens=prompt,
                           requested_output_tokens=output, payload=payload,
                           input_sha256=identity(payload)))
    return result


def frozen_identity(workload):
    items = [dict(id=r['id'], input_sha256=r['input_sha256'], prompt_tokens=r['prompt_tokens'],
                  requested_output_tokens=r['requested_output_tokens']) for r in workload]
    return dict(requests=items, sha256=identity(items))


def parse_metrics(text):
    values = {}
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name = re.split(r'[\s{]', line, maxsplit=1)[0]
        if name not in METRIC_NAMES:
            continue
        match = METRIC_LINE.fullmatch(line)
        if match is None:
            raise ValueError('Malformed required metric: ' + name)
        family, labels, raw = match.groups()
        canonical_labels = []
        if labels:
            content, position = labels[1:-1], 0
            for label in LABEL.finditer(content):
                gap = content[position:label.start()].strip()
                if gap != (',' if canonical_labels else ''):
                    raise ValueError('Malformed metric labels: ' + family)
                canonical_labels.append((label[1], json.loads('"' + label[2] + '"')))
                position = label.end()
            if content[position:].strip() or len({k for k, _ in canonical_labels}) != len(canonical_labels):
                raise ValueError('Malformed metric labels: ' + family)
        series = (family, tuple(sorted(canonical_labels)))
        if series in seen:
            raise ValueError('Duplicate required metric series: ' + family)
        seen.add(series)
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError('Invalid required metric: ' + family)
        key = METRIC_NAMES[family]
        if key != 'cache_usage' and not number.is_integer():
            raise ValueError('Noninteger request/counter metric: ' + family)
        values[key] = max(values.get(key, 0), number) if key == 'cache_usage' else values.get(key, 0) + number
    missing = set(METRIC_NAMES.values()) - values.keys()
    if missing:
        raise ValueError('Missing required metrics: ' + ','.join(sorted(missing)))
    return values


def fetch_metrics(base):
    # Serve streams have a long request deadline; monitoring must fail promptly.
    headers = {}
    key = os.environ.get('VLLM_API_KEY', '')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(base.rstrip('/') + '/metrics', headers=headers)
    with urllib.request.urlopen(req, timeout=3) as response:
        text = response.read().decode()
    values = parse_metrics(text)
    raw = [line for line in text.splitlines()
           if re.split(r'[\s{]', line, maxsplit=1)[0] in METRIC_NAMES]
    return dict(captured_at=captured_at(), values=values, raw=raw)


def settled_metrics(base, settle_seconds):
    if not finite_positive(settle_seconds) or settle_seconds < 1:
        raise ValueError('Metric settling requires at least one finite second')
    # Logger counters lag request execution; this pause must exceed its configured interval.
    time.sleep(settle_seconds)
    first = fetch_metrics(base)
    time.sleep(POLL_INTERVAL)
    second = fetch_metrics(base)
    keys = ('drafts', 'draft_tokens', 'accepted_tokens', 'preemptions')
    if (any(s['values']['running'] or s['values']['waiting'] for s in (first, second))
            or any(first['values'][k] != second['values'][k] for k in keys)):
        raise ValueError('Metrics did not settle to two idle, stable counter readings')
    return dict(second, settle_seconds=settle_seconds, idle_stability_samples=[first, second])


class MetricsPoller:
    def __init__(self, base):
        self.base = base
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(fetch_metrics(self.base))
            except (OSError, ValueError, http.client.HTTPException) as error:
                self.errors.append('metrics_poll:' + type(error).__name__)
            finally:
                self.ready.set()
            self.stop_event.wait(POLL_INTERVAL)

    def start(self):
        self.thread.start()
        if not self.ready.wait(4):
            self.errors.append('metrics_poll_start_timeout')

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            self.errors.append('metrics_poll_stop_timeout')


def finite_positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def summarize_repeat(workload, results, wall_seconds, before, after, samples, instrumentation_errors):
    errors = []
    expected = {r['id']: r for r in workload}
    ids = [r.get('id') for r in results]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != expected.keys():
        errors.append('request_identity_set_mismatch')
    if not finite_positive(wall_seconds):
        errors.append('invalid_wall_time')
    for row in results:
        item = expected.get(row.get('id'))
        usage = row.get('usage') if isinstance(row.get('usage'), dict) else {}
        if item is None:
            continue
        if (not row.get('complete') or row.get('error') is not None or not row.get('stream_done')
                or row.get('finish_reason') != 'length'):
            errors.append(row['id'] + ':incomplete_stream')
        if (type(row.get('output_tokens')) is not int or row['output_tokens'] != item['requested_output_tokens']
                or type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] != item['prompt_tokens']
                or type(usage.get('completion_tokens')) is not int
                or usage['completion_tokens'] != item['requested_output_tokens']):
            errors.append(row['id'] + ':invalid_token_counts')
        ttft, total = row.get('ttft_seconds'), row.get('total_seconds')
        if (not finite_positive(ttft) or not finite_positive(total) or ttft > total
                or (finite_positive(wall_seconds) and total > wall_seconds + 0.001)):
            errors.append(row['id'] + ':invalid_request_times')
        text = row.get('output_text')
        if (not isinstance(text, str) or not text
                or row.get('output_sha256') != hashlib.sha256(text.encode()).hexdigest()
                or row.get('input_sha256') != item['input_sha256']):
            errors.append(row['id'] + ':invalid_payload_hash')
    metric_errors = list(instrumentation_errors)
    required = set(METRIC_NAMES.values())
    snapshots = [before, after] + [s.get('values') for s in samples]
    if not samples:
        metric_errors.append('missing_workload_samples')
    for values in snapshots:
        if (not isinstance(values, dict) or not required.issubset(values)
                or any(type(values[k]) not in (int, float) or not math.isfinite(values[k]) or values[k] < 0
                       for k in required)):
            metric_errors.append('invalid_metrics_snapshot')
            break
    delta = None
    if not metric_errors:
        delta = {k: after[k] - before[k] for k in ('drafts', 'draft_tokens', 'accepted_tokens', 'preemptions')}
        if any(v < 0 for v in delta.values()):
            metric_errors.append('metrics_counter_reset')
        elif delta['draft_tokens'] == 0 or delta['accepted_tokens'] > delta['draft_tokens']:
            metric_errors.append('invalid_speculation_delta')
    valid_metrics = not metric_errors
    errors.extend(metric_errors)
    complete = not errors
    valid_samples = [s['values'] for s in samples if isinstance(s.get('values'), dict)
                     and required.issubset(s['values'])]
    def peak(key):
        return max((s[key] for s in valid_samples), default=None) if valid_metrics else None
    speculation = dict(delta or {}) if valid_metrics else None
    if speculation is not None:
        speculation['acceptance_rate'] = delta['accepted_tokens'] / delta['draft_tokens']
        speculation['interval_includes_warmups'] = False
    ttfts = sorted(r['ttft_seconds'] for r in results) if complete else []
    return dict(complete=complete, error=None if complete else 'invalid_benchmark_evidence',
                validation_errors=errors, requests=len(results), seconds=wall_seconds,
                total_output_tokens=sum(r['output_tokens'] for r in results) if complete else None,
                output_tokens_per_second=sum(r['output_tokens'] for r in results) / wall_seconds if complete else None,
                median_ttft_seconds=statistics.median(ttfts) if complete else None,
                p95_ttft_seconds=ttfts[math.ceil(0.95 * len(ttfts)) - 1] if complete else None,
                p95_method='nearest_rank',
                aggregate_decode_seconds_per_token=(sum(r['total_seconds'] - r['ttft_seconds'] for r in results)
                                                   / sum(r['output_tokens'] - 1 for r in results)) if complete else None,
                speculation=speculation, frozen_identity=frozen_identity(workload), results=results,
                instrumentation=dict(valid=valid_metrics, errors=metric_errors,
                                     sampling_interval_seconds=POLL_INTERVAL, max_running=peak('running'),
                                     max_waiting=peak('waiting'), max_cache_usage=peak('cache_usage'),
                                     observed_running_gt_one=(peak('running') > 1) if valid_metrics else False,
                                     scope='Sampled gauges, not exact maxima; absence of overlap is not proof of serialization',
                                     samples=samples))


def run_stream(base, item):
    started_at = captured_at()
    result = stream_request(base, item['payload'])
    result.update(id=item['id'], input_sha256=item['input_sha256'],
                  prompt_tokens=item['prompt_tokens'], requested_output_tokens=item['requested_output_tokens'],
                  started_at=started_at, captured_at=captured_at())
    return result


def run_performance(base, workload, concurrency, metrics_settle_seconds=6):
    before, after, metric_errors = None, None, []
    before_raw, after_raw = None, None
    try:
        before_raw = settled_metrics(base, metrics_settle_seconds)
        before = before_raw['values']
    except (OSError, ValueError, http.client.HTTPException) as error:
        metric_errors.append('metrics_before:' + type(error).__name__)
    poller = MetricsPoller(base)
    poller.start()
    started_at = captured_at()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda item: run_stream(base, item), workload))
        wall = time.perf_counter() - started
    finally:
        poller.stop()
    try:
        after_raw = settled_metrics(base, metrics_settle_seconds)
        after = after_raw['values']
    except (OSError, ValueError, http.client.HTTPException) as error:
        metric_errors.append('metrics_after:' + type(error).__name__)
    result = summarize_repeat(workload, results, wall, before, after, poller.samples,
                              metric_errors + poller.errors)
    result.update(mode='performance', concurrency=concurrency, started_at=started_at,
                  captured_at=captured_at(), metrics_before=before_raw, metrics_after=after_raw,
                  metrics_settle_seconds=metrics_settle_seconds,
                  metrics_scope='Global server counters require isolated traffic; settle wait must exceed configured stats interval. Two stable reads alone do not prove freshness.',
                  timing_scope='Client submission through all completed streams; excludes warmup and post-run metrics',
                  throughput_scope='Aggregate output tokens per workload wall second; not per-token timestamp throughput')
    return result


def quality_workload(model):
    strict = copy.deepcopy(cases())
    long = copy.deepcopy(long_cases())
    # Distinct long values prevent a concurrent strict response from passing a long oracle.
    for index, case in enumerate(long):
        old = case['expected']
        if isinstance(old, dict):
            new = dict(label=f'concurrent-long-{index}', value=900000 + index)
            replacements = [(old['label'], new['label']), (str(old['value']), str(new['value']))]
        else:
            new = str(900001 + 137 * index) if case['id'].startswith('long-number') else 'concurrent-violet-9182'
            replacements = [(old, new)]
        case['expected'] = new
        for message in case['messages']:
            for old_text, new_text in replacements:
                message['content'] = message['content'].replace(old_text, new_text)
    result = []
    for case in strict + long:
        payload = dict(model=model, messages=case['messages'], temperature=0, seed=17,
                       max_tokens=256, chat_template_kwargs={'enable_thinking': False})
        if 'tools' in case:
            payload.update(tools=case['tools'], tool_choice='auto')
        result.append(dict(case=case, payload=payload, input_sha256=identity(payload),
                           long_context=case['id'].startswith('long-')))
    return result


def run_quality(base, item):
    started_at = captured_at()
    try:
        with request(base, '/v1/chat/completions', item['payload']) as response:
            row = evaluate(item['case'], json.load(response))
    except (OSError, ValueError, http.client.HTTPException) as error:
        row = evaluate(item['case'], {'error': {}})
        row['error'] = 'request_error:' + type(error).__name__
    tokens = (row.get('usage') or {}).get('prompt_tokens', 0)
    if item['long_context']:
        row['long_context_exercised'] = type(tokens) is int and 36864 <= tokens <= 69632
        row['passed'] = row['passed'] and row['long_context_exercised']
    row.update(mode='quality', id=item['case']['id'], input_sha256=item['input_sha256'],
               started_at=started_at, captured_at=captured_at())
    return row


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default=os.environ.get('VLLM_SERVED_MODEL_NAME', 'qwen3.6:27b-mtp-awq'))
    parser.add_argument('--mode', choices=('performance', 'quality'), required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3, help='Performance repeats; quality always runs 38 cases once')
    parser.add_argument('--requests', type=int, default=8, help='Performance requests per repeat; same work at every concurrency')
    parser.add_argument('--workload', choices=tuple(SHAPES), default='short', help='Performance workload')
    parser.add_argument('--warmup', type=int, default=1, help='Full workload warmup passes at selected concurrency, outside timing')
    parser.add_argument('--metrics-settle-seconds', type=float, default=6,
                        help='Pause before/after timing, exceeding server stats interval; finite and at least one second')
    args = parser.parse_args()
    if args.repeats < 1 or args.requests < 1 or args.warmup < 0:
        parser.error('Positive requests/repeats and nonnegative warmup required')
    if not finite_positive(args.metrics_settle_seconds) or args.metrics_settle_seconds < 1:
        parser.error('Finite metrics settling interval of at least one second required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    # Exclusive creation refuses to destroy any previous arm's evidence.
    with args.output.open('x', encoding='utf-8') as output:
        def emit(row):
            row.update(label=args.label)
            output.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
            output.flush()
            print(json.dumps({k: row.get(k) for k in ('mode', 'label', 'repeat', 'id', 'complete', 'passed', 'error')}), flush=True)
        if args.mode == 'quality':
            work = quality_workload(args.model)
            frozen = dict(requests=[dict(id=r['case']['id'], input_sha256=r['input_sha256']) for r in work])
            frozen['sha256'] = identity(frozen['requests'])
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for row in pool.map(lambda item: run_quality(args.base_url, item), work):
                    row.update(concurrency=args.concurrency, frozen_identity=frozen)
                    emit(row)
                    failed |= not row['passed'] or not row['complete']
        else:
            with request(args.base_url, '/tokenize', dict(model=args.model, prompt=SEED_TEXT,
                                                         add_special_tokens=False)) as response:
                seed = json.load(response)['tokens']
            try:
                work = freeze_workload(seed, args.model, args.workload, args.requests)
            except ValueError as error:
                parser.error(str(error))
            warmup_results = []
            for warmup_index in range(args.warmup):
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    warm_rows = list(pool.map(lambda item: run_stream(args.base_url, item), work))
                for item, row in zip(work, warm_rows):
                    row['warmup_pass'] = warmup_index
                    warmup_results.append(row)
                    usage = row.get('usage') or {}
                    if not row['complete'] or usage.get('prompt_tokens') != item['prompt_tokens']:
                        emit(dict(mode='performance', complete=False, error='warmup_failed', warmup_result=row,
                                  output_tokens_per_second=None, frozen_identity=frozen_identity(work)))
                        raise SystemExit(1)
            for repeat in range(args.repeats):
                row = run_performance(args.base_url, work, args.concurrency, args.metrics_settle_seconds)
                row.update(repeat=repeat, workload=args.workload, warmup_passes=args.warmup,
                           warmup_results=warmup_results, kind='screening_not_statistical_qualification')
                emit(row)
                failed |= not row['complete']
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
