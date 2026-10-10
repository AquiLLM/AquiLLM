"""Coordinator-only serial runner; frozen data and verified switch helper."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/exouser/AquiLLM-h100')
CONTAINER = 'compose-vllm-1'


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def wait_ready(image):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        current = json.loads(run('docker', 'inspect', CONTAINER))[0]
        if current['Image'] != image:
            raise RuntimeError('Unexpected running image')
        if not current['State']['Running']:
            raise RuntimeError('Main inference container stopped during startup')
        if current['State'].get('Health', {}).get('Status') == 'healthy':
            return
        time.sleep(5)
    raise RuntimeError('Readiness deadline exceeded')


def measure(label, candidate):
    output = f'/tmp/h100-prefill-ab-{label}.jsonl'
    if Path(output).exists():
        raise RuntimeError('Refusing to overwrite a completed frozen block: ' + output)
    run('docker', 'cp', str(ROOT / 'scripts/h100_performance/serve_bench.py'), CONTAINER + ':/tmp/serve_bench.py')
    with Path(output.replace('.jsonl', '.log')).open('w') as log:
        subprocess.run(['docker', 'exec', CONTAINER, 'python3', '/tmp/serve_bench.py',
                        '--prompt-tokens', '512,8192,32768,36864', '--output-tokens', '256',
                        '--repeats', '10', '--warmup', '1', '--label', label, '--output', output],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    run('docker', 'cp', CONTAINER + ':' + output, output)
    metrics = output.replace('.jsonl', '-' + label + '-metrics.json')
    run('docker', 'cp', CONTAINER + ':' + metrics, metrics)
    rows = [json.loads(line) for line in Path(output).read_text().splitlines()]
    if len(rows) != 44 or any(not row.get('complete') or row.get('error') for row in rows):
        raise RuntimeError('Incomplete or failed serving block')
    logs = run('docker', 'logs', CONTAINER)
    activation = [line for line in logs.splitlines() if 'AQUILLM_H100' in line]
    Path(output.replace('.jsonl', '-activation.json')).write_text(json.dumps(activation, indent=2))
    if candidate and not any('route_exercised' in line and 'prefill' in line for line in activation):
        raise RuntimeError('Prefill candidate was not exercised')
    print(json.dumps({'completed': label, 'requests': len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-image', required=True)
    parser.add_argument('--blocks', default='2,3')
    args = parser.parse_args()
    if socket.gethostname() != 'aquillm-dev2':
        raise RuntimeError('Only the authorized development host is eligible')
    state = json.loads((Path.home() / '.config/aquillm/h100-performance/baseline.json').read_text())
    switch = str(ROOT / 'scripts/h100_performance/dev_switch.py')
    try:
        for block in [int(item) for item in args.blocks.split(',')]:
            print(run('python3', switch, 'rollback'), flush=True)
            wait_ready(state['image'])
            measure(f'baseline-block{block}', False)
            print(run('python3', switch, 'switch', '--image', args.candidate_image,
                      '--mtp', 'baseline', '--prefill', '1'), flush=True)
            wait_ready(args.candidate_image)
            measure(f'prefill-block{block}', True)
    except BaseException:
        print(run('python3', switch, 'rollback'), flush=True)
        wait_ready(state['image'])
        raise


if __name__ == '__main__':
    main()
