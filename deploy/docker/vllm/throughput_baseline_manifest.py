"""Check that a plugin-only candidate preserves every distribution and plugin.

This uses the proven baseline-preservation recipe from the native GDN experiment;
the mixed-gate correctness backport installs no dependency updates.
"""
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys

EXPECTED = {'vllm':'0.23.1rc1.dev748+g2dfaae752', 'torch':'2.11.0+cu130',
    'triton':'3.6.0', 'flashinfer-python':'0.6.13', 'flashinfer-cubin':'0.6.13',
    'flashinfer-jit-cache':'0.6.13+cu130', 'apache-tvm-ffi':'0.1.9',
    'nvidia-cutlass-dsl':'4.5.2', 'nvidia-cutlass-dsl-libs-base':'4.5.2',
    'nvidia-cutlass-dsl-libs-cu13':'4.5.2', 'aquillm-vllm-h100':'0.1.0'}
GENESIS_COMMIT = '34e269301cc3df71ae4b0da00a0a159b16b4e5d8'


def capture():
    return dict(packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in md.distributions()},
        plugins=sorted((e.group,e.name,e.value) for d in md.distributions()
                       for e in d.entry_points if e.group.startswith('vllm.')),
        genesis=subprocess.check_output(['git','-C','/opt/genesis','rev-parse','HEAD'],text=True).strip())


def validate_baseline(current):
    if current['genesis'] != GENESIS_COMMIT:
        raise ValueError('Unsupported baseline Genesis commit')
    for name, version in EXPECTED.items():
        if current['packages'].get(name) != version:
            raise ValueError('Unsupported baseline distribution: '+name)


def main(phase):
    root = Path('/opt/throughput-baseline-evidence')
    root.mkdir(exist_ok=True)
    current = json.loads(json.dumps(capture()))
    validate_baseline(current)
    if phase == 'after':
        if current != json.loads((root/'before.json').read_text()):
            raise ValueError('Plugin-only build changed distribution, plugin or Genesis identity')
    elif phase != 'before':
        raise ValueError('Expected before or after')
    (root/(phase+'.json')).write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,packages_checked=len(current['packages']),
                         plugins=current['plugins'],genesis=current['genesis'])))


if __name__ == '__main__':
    main(sys.argv[1])
