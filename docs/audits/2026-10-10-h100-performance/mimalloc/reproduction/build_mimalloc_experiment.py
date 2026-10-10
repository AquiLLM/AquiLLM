import json
from pathlib import Path
import socket
import subprocess

assert socket.gethostname() == 'aquillm-dev2'
root = '/home/exouser/AquiLLM'
checkout = Path('/home/exouser/AquiLLM-mimalloc-validation')
revision = 'fedc29373c634fd47cf1f4429d133edaa2bc7e91'
base = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

print(run('git','-C',root,'fetch','origin','codex/mimalloc-runtime'),flush=True)
assert run('git','-C',root,'rev-parse','origin/codex/mimalloc-runtime').strip() == revision
if not checkout.exists():
    print(run('git','-C',root,'worktree','add','--detach',str(checkout),revision),flush=True)
assert run('git','-C',str(checkout),'rev-parse','HEAD').strip() == revision
assert not run('git','-C',str(checkout),'status','--porcelain').strip()
subprocess.run(['docker','build','--build-arg','BASE_IMAGE='+base,
                '-f','/tmp/Dockerfile.mimalloc-experiment','-t','aquillm-h100:mimalloc-test',str(checkout)],check=True)
image = run('docker','image','inspect','--format','{{.Id}}','aquillm-h100:mimalloc-test').strip()
record = dict(image=image,base_image=base,source_commit=revision,pr='https://github.com/AquiLLM/AquiLLM/pull/240')
Path('/tmp/h100-mimalloc-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
