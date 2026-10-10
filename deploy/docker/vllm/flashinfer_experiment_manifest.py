"""Build evidence: reject unintended dependency or plugin changes."""
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys

EXPECTED = {
    'flashinfer-python': '0.6.18', 'flashinfer-cubin': '0.6.18',
    'flashinfer-jit-cache': '0.6.18+cu130', 'nvidia-cutlass-dsl': '4.6.2',
    'nvidia-cutlass-dsl-libs-base': '4.6.2', 'nvidia-cutlass-dsl-libs-core': '4.6.2',
    'nvidia-cutlass-dsl-libs-cu12': '4.6.2', 'nvidia-cutlass-dsl-libs-cu13': '4.6.2',
    'nvidia-cuda-nvdisasm': '13.4.92', 'cuda-tile': '1.4.0', 'nccl4py': '0.3.1',
}


def capture():
    return {
        'packages': {d.metadata['Name'].lower().replace('_', '-'): d.version for d in md.distributions()},
        'plugins': sorted((e.group, e.name, e.value) for d in md.distributions() for e in d.entry_points if e.group.startswith('vllm.')),
        'genesis': subprocess.check_output(['git', '-C', '/opt/genesis', 'rev-parse', 'HEAD'], text=True).strip(),
    }


def validate(before, after):
    if before['plugins'] != after['plugins'] or before['genesis'] != after['genesis']:
        raise ValueError('Serving plugin identity changed')
    for name in before['packages'].keys() | after['packages'].keys():
        expected = EXPECTED.get(name, before['packages'].get(name))
        if after['packages'].get(name) != expected:
            raise ValueError(f'Unexpected distribution change: {name}')
    for name, expected in EXPECTED.items():
        if after['packages'].get(name) != expected:
            raise ValueError(f'Candidate dependency missing: {name}')


if __name__ == '__main__':
    root = Path('/opt/flashinfer-upgrade-evidence')
    root.mkdir(exist_ok=True)
    phase = sys.argv[1]
    current = capture()
    # JSON round-trip normalizes entrypoint tuples before comparison.
    current = json.loads(json.dumps(current))
    if phase == 'after':
        validate(json.loads((root/'before.json').read_text()), current)
    elif phase != 'before':
        raise ValueError('Expected before or after')
    (root/f'{phase}.json').write_text(json.dumps(current, indent=2)+'\n')
    print(json.dumps({'phase': phase, 'plugins': current['plugins'], 'genesis': current['genesis']}))
