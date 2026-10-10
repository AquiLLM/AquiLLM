"""Build only the thin web experiment image after the isolated model run."""
import json
from pathlib import Path
import socket
import subprocess

assert socket.gethostname()=='aquillm-dev2'
assert json.loads(Path('/tmp/h100-allocator-api-complete.json').read_text())['completed']
BASE='sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8'
ALLOCATOR='sha256:dc14ba6ec72907fdcbc097a08eb99d69f104d9817a6e5758d29b5c819694926c'
def inspect(kind,target):
    return json.loads(subprocess.check_output(['docker',kind,'inspect',target],text=True))[0]
base=inspect('image',BASE)
assert base['Config']['Entrypoint'] is None
assert base['Config']['Cmd']==['sh','/app/deploy/scripts/run.sh']
assert inspect('container','compose-web-1')['Image']==BASE
assert not Path('/tmp/h100-web-mimalloc-build.json').exists()
with Path('/tmp/h100-web-mimalloc-build.log').open('w') as log:
    subprocess.run(['docker','build','--build-arg','ALLOCATOR_IMAGE='+ALLOCATOR,
        '--build-arg','WEB_IMAGE='+BASE,'-t','aquillm-web:mimalloc-test','-'],
        input=Path('/tmp/Dockerfile.web-mimalloc-experiment').read_text(),text=True,
        stdout=log,stderr=subprocess.STDOUT,check=True)
candidate=inspect('image','aquillm-web:mimalloc-test')
value=dict(image=candidate['Id'],base_image=BASE,allocator_source_image=ALLOCATOR,
           source_commit='fedc29373c634fd47cf1f4429d133edaa2bc7e91',
           pr='https://github.com/AquiLLM/AquiLLM/pull/240')
Path('/tmp/h100-web-mimalloc-build.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
