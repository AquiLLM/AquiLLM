import argparse
import json
from pathlib import Path
import re
import subprocess
import socket

assert socket.gethostname() == 'aquillm-dev2'
out = Path('/tmp/h100-fi-upgrade')
parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Recorded immutable image ID')
parser.add_argument('--source-commit', required=True, help='Full commit from the recorded build identity')
parser.add_argument('--native-launch', action='store_true')
parser.add_argument('--prefix', default='native-parameter-reprobe')
args = parser.parse_args()
if (not re.fullmatch(r'sha256:[a-f0-9]{64}', args.image)
        or not re.fullmatch(r'[a-f0-9]{40}', args.source_commit)
        or not re.fullmatch(r'[a-z0-9-]+', args.prefix)):
    parser.error('Invalid immutable identity or evidence prefix')
image = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',args.image],text=True).strip()
assert image == args.image
name = 'native_parameter_failure.py' if args.native_launch else 'parameter_probe.py'
dest = out/(args.prefix+('-launch.json' if args.native_launch else '-dlpack.json'))
if dest.exists():
    raise RuntimeError('Refusing to overwrite experiment evidence')
result = subprocess.check_output(['docker','run','--rm','--gpus','all','--entrypoint','python3',
    '-v',str(out)+':/evidence:ro',image,'/evidence/'+name],text=True)
record = dict(image=image,source_commit=args.source_commit,
    source_commit_provenance='Operator supplied from recorded build identity',
    probe=json.loads(result))
with dest.open('x') as capture:
    json.dump(record,capture,indent=2)
    capture.write('\n')
print(json.dumps(record,indent=2))
