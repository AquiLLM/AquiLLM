"""Bracket one synthetic model request with the configured worker profiler.

This is an attribution probe, never a latency benchmark. The server must be
started with the matching bounded profiler configuration before calling it.
"""
import argparse
import json
from pathlib import Path

from quality_bench import identity
from serve_bench import request, stream_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen3.6:27b-mtp-awq')
    parser.add_argument('--kind', choices=('decode', 'prefill'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite profile evidence')
    seed_text = 'A synthetic performance note: the project has twelve numbered stages. Summarize the stages and explain their order. '
    with request(args.base_url, '/tokenize', dict(model=args.model, prompt=seed_text,
                                               add_special_tokens=False)) as response:
        seed = json.load(response)['tokens']
    length, outputs = (512, 1024) if args.kind == 'decode' else (36864, 32)
    ids = (seed * (length // len(seed) + 1))[:length]
    payload = dict(model=args.model, prompt=ids, max_tokens=outputs, temperature=0,
                   seed=17, ignore_eos=True, stream=True, stream_options={'include_usage': True})
    record = dict(kind=args.kind, input_sha256=identity(payload), performance_claim=False,
                  prompt_tokens=length, requested_output_tokens=outputs)
    record['warmup'] = stream_request(args.base_url, payload)
    if not record['warmup']['complete']:
        raise SystemExit('Unprofiled warmup failed; profiling not started')
    try:
        with request(args.base_url, '/start_profile', {}) as response:
            record['start_status'] = response.status
        record['profiled'] = stream_request(args.base_url, payload)
    finally:
        try:
            with request(args.base_url, '/stop_profile', {}) as response:
                record['stop_status'] = response.status
        except Exception as exc:
            record['stop_error'] = type(exc).__name__
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    valid = (record.get('profiled', {}).get('complete') and
             record.get('start_status') == 200 and record.get('stop_status') == 200)
    print(json.dumps({'kind':args.kind, 'valid':bool(valid), 'performance_claim':False}), flush=True)
    if not valid:
        raise SystemExit('Profile request or profiler lifecycle failed')


if __name__ == '__main__':
    main()
