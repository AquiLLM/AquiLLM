"""Serialized development-only allocator comparison with verified prefill rollback."""
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone
from run_prefill_blocks import run, wait_ready

assert socket.gethostname() == 'aquillm-dev2'
ROOT=Path('/home/exouser/AquiLLM-h100')
C='compose-vllm-1'
IMAGE='sha256:dc14ba6ec72907fdcbc097a08eb99d69f104d9817a6e5758d29b5c819694926c'
PREFILL='sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
SWITCH=str(ROOT/'scripts/h100_performance/allocator_switch.py')

def capture(name, allocator):
    try:
        value=json.loads(run('docker','exec',C,'python3','/tmp/probe_serving_allocator.py',allocator))
    except subprocess.CalledProcessError as error:
        print(error.output,flush=True)  # This probe emits only public allocator state.
        raise
    Path('/tmp/'+name+'.json').write_text(json.dumps(value,indent=2)+'\n')
    return value

def quality(label):
    for script,kind,count in [('quality_bench.py','strict',32),('long_quality_bench.py','long',6)]:
        path=f'/tmp/h100-allocator-{label}-{kind}.jsonl'
        assert not Path(path).exists()
        with Path(path+'.log').open('w') as log:
            subprocess.run(['docker','exec',C,'python3','/tmp/h100_performance/'+script,
                            '--label',label,'--output',path],stdout=log,stderr=subprocess.STDOUT,check=True)
        run('docker','cp',C+':'+path,path)
        rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
        assert len(rows)==count and all(r['passed'] and r['complete'] and not r.get('error') for r in rows)
        print(json.dumps({'quality':label+'-'+kind,'passed':len(rows)}),flush=True)

def measure(label):
    path=f'/tmp/h100-allocator-{label}.jsonl'
    assert not Path(path).exists()
    with Path(path+'.log').open('w') as log:
        subprocess.run(['docker','exec',C,'python3','/tmp/h100_performance/serve_bench.py',
                        '--prompt-tokens','512,8192,32768,36864','--output-tokens','256',
                        '--repeats','10','--warmup','1','--label',label,'--output',path],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    run('docker','cp',C+':'+path,path)
    metrics=path.replace('.jsonl','-'+label+'-metrics.json')
    run('docker','cp',C+':'+metrics,metrics)
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    assert len(rows)==44 and all(r['complete'] and not r.get('error') for r in rows)
    logs=run('docker','logs','--timestamps',C)
    proof=[line for line in logs.splitlines() if any(k in line for k in ('AQUILLM_H100','aquillm: allocator=','aquillm: mimalloc_version='))]
    assert any('route_exercised prefill' in line for line in proof)
    Path(path.replace('.jsonl','-activation.json')).write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'completed':label,'requests':len(rows),'at':datetime.now(timezone.utc).isoformat()}),flush=True)

def others():
    return {line.split()[0]:line.split()[1] for line in run('docker','ps','--format','{{.Names}} {{.ID}}').splitlines()
            if not line.startswith(C+' ')}

before=others()
try:
    for block in (1,2,3):
        for allocator in ('system','mimalloc'):
            label=f'{allocator}-block{block}'
            print(run('python3',SWITCH,'switch','--allocator',allocator),flush=True)
            wait_ready(IMAGE)
            assert others()==before
            run('docker','cp',str(ROOT/'scripts/h100_performance'),C+':/tmp/h100_performance')
            run('docker','cp','/tmp/probe_serving_allocator.py',C+':/tmp/probe_serving_allocator.py')
            capture('h100-allocator-'+label+'-before',allocator)
            if block==1:
                quality(label)
            measure(label)
            capture('h100-allocator-'+label+'-after',allocator)
            assert others()==before
    Path('/tmp/h100-allocator-api-complete.json').write_text(json.dumps({'completed':True,'at':datetime.now(timezone.utc).isoformat()})+'\n')
finally:
    print(run('python3',SWITCH,'rollback'),flush=True)
    wait_ready(PREFILL)
    assert others()==before
    print(json.dumps({'restored_prefill_image':PREFILL,'healthy':True}),flush=True)
