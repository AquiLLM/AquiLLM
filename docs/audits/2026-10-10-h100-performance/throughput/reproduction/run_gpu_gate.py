"""Run the bounded real-kernel gate with the serving baseline idle and healthy."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT = Path('/home/exouser/AquiLLM-throughput')
OUT = Path('/tmp/h100-throughput')
BASE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'


def run(*args):
    return subprocess.check_output(args,text=True,stderr=subprocess.PIPE).strip()


def main():
    assert socket.gethostname() == 'aquillm-dev2'
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    assert re.fullmatch('sha256:[0-9a-f]{64}',args.image)
    assert re.fullmatch('[a-z0-9-]+',args.prefix)
    lock=(OUT/'execution.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    current=json.loads(run('docker','inspect','compose-vllm-1'))[0]
    assert current['Image']==BASE and current['State']['Health']['Status']=='healthy'
    # Check actual request gauges without printing or forwarding the API secret.
    idle_script="""import os,urllib.request
r=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
t=urllib.request.urlopen(r,timeout=5).read().decode()
rows=[x for x in t.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
assert len(rows)==2 and all(float(x.rsplit(' ',1)[1])==0 for x in rows)
"""
    run('docker','exec','compose-vllm-1','python3','-c',idle_script)
    env=[]
    for item in current['Config']['Env']:
        if item.startswith(('GENESIS_','AQUILLM_H100_','VLLM_MODEL=')):
            env += ['--env',item]
    name='aquillm-gdn-gate-'+args.prefix
    existing=subprocess.run(['docker','container','inspect',name],capture_output=True)
    assert existing.returncode != 0, 'Test container name already exists'
    command=['docker','run','--name',name,'--rm','--gpus','device=0','--ipc=host',
        '--entrypoint','python3',*env,'-v',str(ROOT)+':/workspace:ro',
        '-v',str(OUT)+':/evidence',args.image,'/evidence/gdn_gate.py']
    path=OUT/(args.prefix+'-gpu.log')
    started=time.time()
    error=None
    rc=1
    try:
        with path.open('x') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        # Stop the GPU process even if a timeout/interruption killed only docker's client.
        subprocess.run(['docker','rm','--force',name],capture_output=True,timeout=30)
        assert subprocess.run(['docker','container','inspect',name],capture_output=True).returncode != 0
    restored=json.loads(run('docker','inspect','compose-vllm-1'))[0]
    result=dict(image=args.image,returncode=rc,error=error,wall_seconds=time.time()-started,
        baseline_unchanged=restored['Id']==current['Id'] and restored['Image']==BASE,
        baseline_healthy=restored['State']['Health']['Status']=='healthy',
        source_commit=run('git','-C',str(ROOT),'rev-parse','HEAD'))
    (OUT/(args.prefix+'-gpu.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(rc if rc else int(bool(error) or not result['baseline_unchanged']
                                      or not result['baseline_healthy']))


if __name__=='__main__':
    main()
