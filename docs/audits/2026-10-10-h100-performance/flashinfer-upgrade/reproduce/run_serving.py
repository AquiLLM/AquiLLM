"""Serialized development comparison; always restore captured prefill deployment."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path('/home/exouser/AquiLLM-flashinfer')
OUT = Path('/tmp/h100-fi-upgrade')
STATE = Path('/home/exouser/.config/aquillm/flashinfer-upgrade')
CONTAINER = 'compose-vllm-1'


def run(*args):
    # Join the complete operation before rollback, including Compose children.
    process = subprocess.Popen(args, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate()
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        raise RuntimeError('Operation failed: '+str(args[:2]))
    return output


def inspect():
    return json.loads(run('docker','inspect',CONTAINER))[0]


def others():
    return sorted(line for line in run('docker','ps','--format','{{.Names}} {{.ID}}').splitlines() if not line.startswith(CONTAINER+' '))


def wait_ready(image):
    started = None
    for _ in range(240):
        state = inspect()
        if state['Image'] != image:
            raise RuntimeError('Unexpected image while waiting for health')
        if state['State'].get('Health',{}).get('Status') == 'healthy':
            return
        if not state['State'].get('Running'):
            raise RuntimeError('Model container exited')
        current_start = state['State']['StartedAt']
        if started is not None and current_start != started:
            raise RuntimeError('Model container restarted during startup')
        started = current_start
        time.sleep(5)
    raise RuntimeError('Model health deadline exceeded')


def switch(image, profile, gdn):
    before = inspect()['Id']
    if profile == gdn == 'baseline':
        print(run('python3',str(ROOT/'scripts/h100_performance/dev_switch.py'),'rollback',
                  '--state-dir',str(STATE)),flush=True)
    else:
        print(run('python3',str(ROOT/'scripts/h100_performance/dev_switch.py'),'switch',
              '--state-dir',str(STATE),'--image',image,'--prefill','1','--mtp','baseline',
              '--runtime-profile',profile,'--gdn',gdn),flush=True)
    wait_ready(image)
    # Restart if Compose reused the previous block's image/settings.
    if inspect()['Id'] == before:
        run('docker','restart',CONTAINER)
        wait_ready(image)
    return inspect()['State']['StartedAt']


def execute(script, label, suffix, *extra):
    filename = label+'-'+suffix+'.jsonl'
    dest = OUT/filename
    if dest.exists(): raise RuntimeError('Refusing to overwrite experiment evidence')
    inside = '/tmp/'+filename
    with (OUT/(filename+'.log')).open('w') as log:
        result = subprocess.run(['docker','exec',CONTAINER,'python3','/tmp/fi_h100/'+script,
              '--label',label,'--output',inside,*extra],stdout=log,stderr=subprocess.STDOUT)
    try:
        run('docker','cp',CONTAINER+':'+inside,str(dest))
    except subprocess.CalledProcessError:
        raise RuntimeError('Benchmark did not produce evidence') from None
    rows = [json.loads(x) for x in dest.read_text().splitlines()]
    if result.returncode or not rows or any(not x.get('complete') or x.get('error') for x in rows):
        raise RuntimeError('Invalid benchmark response: '+label+' '+suffix)
    if any(row.get('label') != label for row in rows):
        raise RuntimeError('Benchmark label mismatch')
    if script in ('quality_bench.py','long_quality_bench.py'):
        expected = 32 if script == 'quality_bench.py' else 6
        if len(rows) != expected or any(not x.get('passed') for x in rows):
            raise RuntimeError('Quality gate failed: '+label+' '+suffix)
    elif script == 'serve_bench.py':
        metrics = dest.stem+'-'+label+'-metrics.json'
        run('docker','cp',CONTAINER+':/tmp/'+metrics,str(OUT/metrics))
    elif script == 'load_bench.py':
        if [(r.get('concurrency'),r.get('requests')) for r in rows] != [(2,4),(4,8)]:
            raise RuntimeError('Incomplete queued-load matrix')
        if any(len(r.get('results',[])) != r['requests'] or
               any(not x.get('complete') or x.get('output_tokens')!=256 for x in r['results']) for r in rows):
            raise RuntimeError('Incomplete queued-load responses')
        if any(not math.isfinite(r[k]) or r[k] <= 0 for r in rows for k in ('seconds','output_tokens_per_second')):
            raise RuntimeError('Invalid queued-load timing')
    elif script == 'cancel_bench.py':
        if len(rows) != 2 or any(not r.get('passed') for r in rows):
            raise RuntimeError('Client-close recovery gate failed')
    print(json.dumps({'label':label,'stage':suffix,'valid':len(rows)}),flush=True)


def capture_routes(label, gdn, since, expected_key):
    log = run('docker','logs','--since',since,CONTAINER)
    lines = [x for x in log.splitlines() if 'AQUILLM_H100' in x]
    (OUT/(label+'-activation.json')).write_text(json.dumps(lines,indent=2)+'\n')
    if not any('route_exercised' in x and 'prefill' in x and 'runtime='+expected_key in x for x in lines):
        raise RuntimeError('Existing prefill route was not exercised')
    if gdn != 'baseline' and not any('route_exercised' in x and ('gdn='+gdn) in x for x in lines):
        raise RuntimeError('Candidate GDN route was not exercised')


def main():
    assert socket.gethostname() == 'aquillm-dev2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-image', required=True)
    parser.add_argument('--gdn',choices=('baseline','flashinfer','native-fp16'),default='baseline')
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--blocks',type=int,default=1)
    parser.add_argument('--candidate-first',action='store_true')
    parser.add_argument('--fixed-order',action='store_true',help='Baseline then candidate in every paired block')
    parser.add_argument('--repeats',type=int,default=3)
    args = parser.parse_args()
    args.candidate_image = json.loads(run('docker','image','inspect',args.candidate_image))[0]['Id']
    import sys
    sys.path.insert(0,str(ROOT/'deploy/vllm_plugins/h100_kernels/src'))
    from aquillm_vllm_h100.prefill_profiles import RUNTIME_KEY, CANDIDATE_RUNTIME_KEY
    original = json.loads((STATE/'baseline.json').read_text())
    if inspect()['Image'] != original['image']:
        raise RuntimeError('Experiment must begin on the captured baseline image')
    if args.blocks < 1 or args.repeats < 1:
        raise ValueError('Positive block/repeat counts required')
    candidate_profile = 'native-gdn-baseline' if args.gdn == 'native-fp16' else 'flashinfer-0.6.18'
    protected = others()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for block in range(1,args.blocks+1):
            arms = [('baseline',original['image'],'baseline','baseline'),
                    ('candidate',args.candidate_image,candidate_profile,args.gdn)]
            if not args.fixed_order and (block%2==0) != args.candidate_first: arms.reverse()
            for arm,image,profile,gdn in arms:
                label = f'{args.prefix}-{arm}-block{block}'
                since = switch(image,profile,gdn)
                if others()!=protected: raise RuntimeError('Unrelated service changed')
                run('docker','cp',str(ROOT/'scripts/h100_performance'),CONTAINER+':/tmp/fi_h100')
                run('docker','cp',str(OUT/'load_bench.py'),CONTAINER+':/tmp/fi_h100/load_bench.py')
                if args.gdn != 'baseline':
                    run('docker','cp',str(OUT/'cancel_bench.py'),CONTAINER+':/tmp/fi_h100/cancel_bench.py')
                for script,suffix in [('quality_bench.py','strict'),('long_quality_bench.py','long')]:
                    execute(script,label,suffix)
                capture_routes(label,gdn,since,RUNTIME_KEY if profile!='flashinfer-0.6.18' else CANDIDATE_RUNTIME_KEY)
                if args.gdn != 'baseline':
                    execute('cancel_bench.py',label,'cancel')
                execute('serve_bench.py',label,'latency','--model','qwen3.6:27b-mtp-awq',
                        '--prompt-tokens','512,8192,32768,36864','--output-tokens','256','--warmup','1','--repeats',str(args.repeats))
                execute('serve_bench.py',label,'decode','--model','qwen3.6:27b-mtp-awq',
                        '--prompt-tokens','512','--output-tokens','1024','--warmup','1','--repeats',str(args.repeats))
                execute('load_bench.py',label,'load')
    except BaseException as error:
        # Preserve startup diagnostics before Compose removes the failed arm.
        # Avoid INFO argument/environment dumps; retain error/trace and route lines.
        failure = {'error_type':type(error).__name__,'error':str(error)}
        try:
            current = inspect()
            failure.update(image=current['Image'],started_at=current['State']['StartedAt'],
                           restart_count=current['RestartCount'])
            log = run('docker','logs',CONTAINER)
            failure['diagnostics'] = [line for line in log.splitlines() if
                any(token in line for token in ('ERROR','Traceback','Error:','Exception:',
                    'SystemExit','AQUILLM_H100'))][-400:]
        except BaseException as diagnostic_error:
            failure['diagnostic_error_type'] = type(diagnostic_error).__name__
        (OUT/(args.prefix+'-failure.json')).write_text(json.dumps(failure,indent=2)+'\n')
        print(json.dumps({'failure':failure.get('error'),'diagnostic_file':args.prefix+'-failure.json'}),flush=True)
        raise
    finally:
        print(run('python3',str(ROOT/'scripts/h100_performance/dev_switch.py'),'rollback','--state-dir',str(STATE)),flush=True)
        wait_ready(original['image'])
        if others()!=protected: raise RuntimeError('Unrelated service identity changed')
        state = inspect()
        proof = {'restored_image':state['Image'],'healthy':True,'unrelated_services_unchanged':True}
        (OUT/(args.prefix+'-restored.json')).write_text(json.dumps(proof,indent=2)+'\n')
        print(json.dumps(proof),flush=True)


if __name__=='__main__': main()
