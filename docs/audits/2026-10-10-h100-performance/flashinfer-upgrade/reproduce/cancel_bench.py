"""Bounded disconnect/recovery screening; does not claim scheduler abort proof."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import urllib.request

from quality_bench import cases, evaluate, identity
from serve_bench import request


def metrics(base, timeout=5):
    headers = {}
    key = os.environ.get('VLLM_API_KEY','')
    if key:
        headers['Authorization'] = 'Bearer '+key
    req = urllib.request.Request(base+'/metrics', headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        lines = response.read().decode().splitlines()
    result = dict(running=0.0, waiting=0.0, generated=0.0, length_finished=0.0)
    found = set()
    names = {'vllm:num_requests_running':'running', 'vllm:num_requests_waiting':'waiting',
             'vllm:generation_tokens_total':'generated', 'vllm:request_success_total':'length_finished'}
    for line in lines:
        if not line or line.startswith('#'):
            continue
        sample, raw = line.rsplit(' ',1)
        name = sample.split('{',1)[0]
        if name not in names:
            continue
        if name == 'vllm:request_success_total' and 'finished_reason="length"' not in sample:
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise RuntimeError('Invalid disconnect-probe metric')
        result[names[name]] += value
        found.add(names[name])
    if not {'running','waiting','generated','length_finished'} <= found:
        raise RuntimeError('Disconnect-probe metrics missing')
    return result


def wait_idle(base, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        current = metrics(base, timeout=max(0.01,min(2,deadline-time.monotonic())))
        if current['running'] == current['waiting'] == 0:
            return current
        time.sleep(min(0.05,max(0,deadline-time.monotonic())))
    raise RuntimeError('Server did not become idle after stream close')


def check(base, model, case):
    payload = dict(model=model, messages=case['messages'], temperature=0, seed=17,
                   max_tokens=256, chat_template_kwargs={'enable_thinking':False})
    with request(base, '/v1/chat/completions', payload) as response:
        result = json.load(response)
    return evaluate(case, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, model = 'http://127.0.0.1:8000', 'qwen3.6:27b-mtp-awq'
    checks = [case for case in cases() if case['id'] in ('number-0', 'multiturn-0')]
    with args.output.open('x') as out:
        for case in checks:
            wait_idle(base)
            before = check(base, model, case)
            counters_before = wait_idle(base)
            payload = dict(model=model, prompt='Write a long numbered list of fictional projects and explain each one. ',
                max_tokens=4096, temperature=0, seed=17, ignore_eos=True, stream=True)
            chunks, done, terminal, request_id = 0, False, False, None
            with request(base, '/v1/completions', payload) as response:
                for raw in response:
                    if not raw.startswith(b'data:'):
                        continue
                    data = raw[5:].strip()
                    if data == b'[DONE]':
                        done = True
                        break
                    item = json.loads(data)
                    if item.get('error'):
                        raise RuntimeError('Cancellation probe server error')
                    request_id = item.get('id', request_id)
                    if any(choice.get('finish_reason') is not None for choice in item.get('choices',[])):
                        terminal = True
                        break
                    if any(choice.get('text') for choice in item.get('choices', [])):
                        chunks += 1
                    if chunks >= 8:
                        break
                active = metrics(base)
            closed_at = time.monotonic()
            counters_after = wait_idle(base, seconds=2)
            idle_after_seconds = time.monotonic()-closed_at
            generated = counters_after['generated']-counters_before['generated']
            length_finished = counters_after['length_finished']-counters_before['length_finished']
            after = [check(base, model, case) for _ in range(2)]
            passed = (before['passed'] and all(row['passed'] for row in after)
                and chunks >= 8 and not done and not terminal and request_id is not None
                and active['running'] > 0 and 0 < generated < 4096
                and length_finished == 0 and idle_after_seconds <= 2)
            row = dict(label=args.label, id=case['id'], complete=True, error=None, passed=passed,
                request_id=request_id, closed_after_nonempty_chunks=chunks, reached_done_before_close=done,
                terminal_before_close=terminal, server_abort_confirmation='not_observed',
                physical_state_slot_reuse='not_observed', running_while_stream_open=active['running'],
                idle_after_seconds=idle_after_seconds, generated_token_delta=generated,
                length_finished_delta=length_finished, counters_before=counters_before,counters_after=counters_after,
                cancellation_input_sha256=identity(payload), before=before, after=after)
            out.write(json.dumps(row)+'\n'); out.flush()
            print(json.dumps({key:row[key] for key in ('label','id','passed','closed_after_nonempty_chunks')}),flush=True)
            if not passed:
                raise RuntimeError('Bounded disconnect/recovery check failed')


if __name__ == '__main__':
    main()
