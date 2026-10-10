import json
from pathlib import Path
import socket
import subprocess
import sys
import re

if socket.gethostname() != 'aquillm-dev2':
    raise RuntimeError('This helper requires aquillm-dev2')
root = Path('/home/exouser/AquiLLM-flashinfer')
out = Path('/tmp/h100-fi-upgrade')
# Validate the mounted source before importing it or making any Docker/GPU call.
build = json.loads((out/'build.json').read_text())
if not isinstance(build, dict) or not isinstance(build.get('commit'), str) or not re.fullmatch('[0-9a-f]{40}', build['commit']):
    raise ValueError('build.json must contain a full immutable commit ID')
if not isinstance(build.get('image'), str) or not re.fullmatch('sha256:[0-9a-f]{64}', build['image']):
    raise ValueError('build.json must contain an immutable image ID')
if not isinstance(build.get('checkout'), str) or Path(build['checkout']).resolve() != root.resolve():
    raise ValueError('build.json must identify the mounted experimental checkout')
if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip():
    raise RuntimeError('Experimental checkout must be clean, including untracked files')
if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != build['commit']:
    raise RuntimeError('Experimental checkout HEAD does not match build.json commit')
sys.path.insert(0,str(root/'scripts/h100_performance'))
from snapshot import SAFE_ENV
current = json.loads(subprocess.check_output(['docker','inspect','compose-vllm-1'],text=True))[0]
values = dict(x.split('=',1) for x in current['Config']['Env'] if '=' in x)
env = {k:v for k,v in values.items() if k in SAFE_ENV or k.startswith(('GENESIS_P','GENESIS_ENABLE_','AQUILLM_H100_')) or k=='GENESIS_ENFORCE_VERSION_RANGE'}
env.update(AQUILLM_H100_RUNTIME_PROFILE='flashinfer-0.6.18',AQUILLM_H100_GDN='baseline',AQUILLM_H100_PREFILL='1',AQUILLM_H100_MTP_KERNEL='baseline')
image = build['image']
baseline_stack = '--baseline-stack' in sys.argv
native_image = '--native-image' in sys.argv
if native_image:
    env['AQUILLM_H100_RUNTIME_PROFILE'] = 'baseline'
if baseline_stack:
    image = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
    env['AQUILLM_H100_RUNTIME_PROFILE'] = 'baseline'
    # Direct-kernel qualification only; the serving adapter stays disabled.
    env['PYTHONPATH'] = '/workspace/deploy/vllm_plugins/h100_kernels/src'
mode = 'ceiling' if '--ceiling' in sys.argv else 'benchmark' if '--benchmark' in sys.argv else 'tests'
suffix = '-stride96' if '--stride96' in sys.argv else ''
args = ['docker','run','--rm','--gpus','all','--ipc','host','--entrypoint','python3',
        '-v',str(root)+':/workspace:ro','-v',str(out)+':/evidence',
        '-v',str(out/'triton-cache')+':/root/.triton',
        '-v',str(out/'flashinfer-cache')+':/root/.cache/flashinfer']
for key,value in env.items(): args += ['-e',key+'='+value]
args += [image,'/evidence/gdn_gate.py']+(['--'+mode] if mode!='tests' else [])
if suffix: args.append('--stride96')
if baseline_stack: args.append('--baseline-stack')
if native_image: args.append('--native-image')
prefix = 'baseline-stack-' if baseline_stack else 'native-baseline-image-' if native_image else ''
if '--evidence-prefix' in sys.argv:
    prefix = sys.argv[sys.argv.index('--evidence-prefix')+1]
    assert re.fullmatch('[a-z0-9-]+',prefix)
logpath = out/(prefix+'fp16-native-gdn-'+mode+suffix+'.log')
with logpath.open('w') as log:
    result = subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
print(logpath.read_text()[-7000:],flush=True)
raise SystemExit(result.returncode)
