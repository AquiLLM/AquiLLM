"""Read-only, filtered graph configuration evidence during a serving arm."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', required=True)
parser.add_argument('--image', required=True)
args = parser.parse_args()
assert re.fullmatch('[a-z0-9-]+', args.prefix)
state = json.loads(subprocess.check_output(['docker', 'inspect', 'compose-vllm-1'], text=True))[0]
assert state['Image'] == args.image
assert state['State'].get('Health', {}).get('Status') == 'healthy'
env = dict(x.split('=', 1) for x in state['Config']['Env'] if '=' in x)
tokens = shlex.split(env.get('VLLM_EXTRA_ARGS', ''))
config = json.loads(tokens[tokens.index('--compilation-config') + 1])
assert set(config) == {'cudagraph_capture_sizes'}
log = subprocess.check_output(['docker', 'logs', '--since', state['State']['StartedAt'],
                               'compose-vllm-1'], text=True, stderr=subprocess.STDOUT)
counts = {}
for line in log.splitlines():
    match = re.search(r'Capturing CUDA graphs \([^)]*, (FULL|PIECEWISE)\).*?\b(\d+)/(\d+)', line)
    if match and match.group(2) == match.group(3):
        counts[match.group(1)] = int(match.group(3))
result = dict(image=state['Image'], started=state['State']['StartedAt'],
              explicit_compilation_config=config, completed_capture_counts=counts,
              scope='Configured sizes and startup capture counts; no per-request replay measurement')
path = Path('/tmp/h100-throughput') / (args.prefix + '-graph-settings.json')
with path.open('x') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps(result))
