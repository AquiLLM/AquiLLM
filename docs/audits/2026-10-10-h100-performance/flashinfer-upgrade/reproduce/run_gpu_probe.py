import json
import os
from pathlib import Path
import socket
import subprocess
import sys

assert socket.gethostname() == 'aquillm-dev2'
root = '/home/exouser/AquiLLM-flashinfer'
sys.path.insert(0, root+'/scripts/h100_performance')
from snapshot import SAFE_ENV
current = json.loads(subprocess.check_output(['docker','inspect','compose-vllm-1'], text=True))[0]
values = dict(x.split('=',1) for x in current['Config']['Env'] if '=' in x)
env = {k:v for k,v in values.items() if k in SAFE_ENV or k.startswith(('GENESIS_P','GENESIS_ENABLE_','AQUILLM_H100_')) or k=='GENESIS_ENFORCE_VERSION_RANGE'}
env.update(AQUILLM_H100_RUNTIME_PROFILE='flashinfer-0.6.18', AQUILLM_H100_GDN='baseline',
           AQUILLM_H100_PREFILL='1', AQUILLM_H100_MTP_KERNEL='baseline', AQUILLM_RUN_FLASHINFER_AUX='1')
image = json.loads(Path('/tmp/h100-fi-upgrade/build.json').read_text())['image']
prefix = 'native-baseline-' if '--native-image' in sys.argv else ''
if prefix:
    env.update(AQUILLM_H100_RUNTIME_PROFILE='native-gdn-baseline', AQUILLM_H100_GDN='native-fp16',
               AQUILLM_EVIDENCE_PREFIX=prefix)
if os.environ.get('AQUILLM_IMPORT_PROBE_ONLY') == '1':
    env['AQUILLM_IMPORT_PROBE_ONLY'] = '1'
args = ['docker','run','--rm','--gpus','all','--ipc','host','--entrypoint','python3',
        '-v',root+':/workspace:ro','-v','/tmp/h100-fi-upgrade:/evidence']
for key,value in env.items(): args += ['-e',key+'='+value]
args += [image, '/evidence/gpu_probe.py']
logpath = Path('/tmp/h100-fi-upgrade')/(prefix+'gpu-probe.log')
with logpath.open('w') as log:
    result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
print(logpath.read_text()[-4000:], flush=True)
raise SystemExit(result.returncode)
