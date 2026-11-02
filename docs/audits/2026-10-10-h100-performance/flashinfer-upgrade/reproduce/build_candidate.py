import argparse
import json
from pathlib import Path
import re
import socket
import subprocess

parser = argparse.ArgumentParser(description='Build a clean, explicitly pinned experimental checkout.')
parser.add_argument('--commit', required=True, help='Full 40-character existing Git commit ID')
parser.add_argument('--native-baseline', action='store_true')
options = parser.parse_args()
if not re.fullmatch('[0-9a-fA-F]{40}', options.commit):
    parser.error('--commit must be a full 40-character hexadecimal commit ID')
revision = options.commit.lower()
if socket.gethostname() != 'aquillm-dev2':
    raise RuntimeError('This helper requires aquillm-dev2')
root = Path('/home/exouser/AquiLLM')
checkout = Path('/home/exouser/AquiLLM-flashinfer')
branch = 'codex/flashinfer-gdn-upgrade'
base = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
evidence = Path('/tmp/h100-fi-upgrade')
if checkout.resolve() == root.resolve():
    raise RuntimeError('Experimental checkout must be separate from the main checkout')

def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def validate_checkout(expected=None):
    if run('git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=all').strip():
        raise RuntimeError('Experimental checkout must be clean, including untracked files')
    if expected is not None and run('git', '-C', str(checkout), 'rev-parse', 'HEAD').strip() != expected:
        raise RuntimeError('Experimental checkout does not match the requested commit')


if checkout.exists():
    validate_checkout()
print(run('git', '-C', str(root), 'fetch', 'origin', branch), flush=True)
resolved = run('git', '-C', str(root), 'rev-parse', '--verify', revision+'^{commit}').strip()
if resolved != revision:
    raise RuntimeError('Requested object must resolve to the exact commit ID')
if not checkout.exists():
    print(run('git', '-C', str(root), 'worktree', 'add', '--detach', str(checkout), revision), flush=True)
else:
    print(run('git', '-C', str(checkout), 'checkout', '--detach', revision), flush=True)
validate_checkout(revision)
native = options.native_baseline
recipe = 'Dockerfile.native-gdn-baseline' if native else 'Dockerfile.flashinfer-experiment'
tag = 'aquillm-h100:native-gdn-test' if native else 'aquillm-h100:flashinfer-0618-test'
command = ['docker', 'build', '--build-arg', 'BASE_IMAGE='+base, '-f', str(checkout/'deploy/docker/vllm'/recipe), '-t', tag, str(checkout)]
with (evidence/'build.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print((evidence/'build.log').read_text()[-12000:], flush=True)
    raise SystemExit(result.returncode)
validate_checkout(revision)
image = run('docker', 'image', 'inspect', '--format', '{{.Id}}', tag).strip()
if not re.fullmatch('sha256:[0-9a-f]{64}', image):
    raise RuntimeError('Docker did not return an immutable image ID')
record = {'image': image, 'base': base, 'commit': revision, 'checkout': str(checkout), 'recipe':recipe}
(evidence/'build.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
