"""Serialized development-only screen/profile with exact baseline recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

ROOT = Path('/home/exouser/AquiLLM-throughput')
OUT = Path('/tmp/h100-throughput')
STATE = Path('/home/exouser/.config/aquillm/throughput-profile')
CONTAINER = 'compose-vllm-1'
IMAGE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'


def run(*args):
    child = subprocess.Popen(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        output, _ = child.communicate(timeout=1800)
    except BaseException:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
        raise
    if child.returncode:
        # Suppress arbitrary subprocess output: Compose can expose environment.
        raise RuntimeError('Operation failed: ' + str(args[:2]))
    return output


def inspect():
    return json.loads(run('docker', 'inspect', CONTAINER))[0]


def others():
    return sorted(line for line in run('docker','ps','--format','{{.Names}} {{.ID}}').splitlines()
                  if not line.startswith(CONTAINER+' '))


def ready():
    started = None
    for _ in range(240):
        state = inspect()
        if state['Image'] != IMAGE:
            raise RuntimeError('Unexpected image')
        if state['State'].get('Health', {}).get('Status') == 'healthy':
            return state
        if not state['State']['Running']:
            raise RuntimeError('Model container stopped')
        now = state['State']['StartedAt']
        if started is not None and started != now:
            raise RuntimeError('Model container restarted during startup')
        started = now
        time.sleep(5)
    raise RuntimeError('Model health deadline exceeded')


def switch(action, limit=1, profile=None):
    cmd = ['python3', str(ROOT/'scripts/h100_performance/throughput_switch.py'),
           action, '--state-dir', str(STATE)]
    if action == 'switch':
        cmd += ['--max-num-seqs', str(limit)]
        if profile:
            cmd += ['--profile', profile]
    print(run(*cmd), flush=True)


def install_probes(mode):
    run('docker','exec',CONTAINER,'mkdir','-p','/tmp/h100-probes')
    names = ['serve_bench.py','quality_bench.py','long_quality_bench.py','profile_trace.py']
    if mode == 'screen':
        names.append('throughput_bench.py')
    for name in names:
        run('docker','cp',str(ROOT/'scripts/h100_performance'/name),CONTAINER+':/tmp/h100-probes/'+name)


def diagnostics(label):
    state = inspect()
    log = run('docker','logs','--since',state['State']['StartedAt'],CONTAINER)
    # Only code diagnostics, not request bodies, environment or argument dumps.
    lines = []
    for line in log.splitlines():
        if ('AQUILLM_H100' in line or
            re.search(r'(?:^|\s)(?:RuntimeError|ValueError|AssertionError|BufferError|torch\.OutOfMemoryError):',line) or
            'CUDA error:' in line or 'illegal memory access' in line):
            if not re.search(r'api.?key|Bearer|password|token=',line,re.I):
                lines.append(line[:2000])
    result = dict(image=state['Image'], running=state['State']['Running'],
                  health=state['State'].get('Health',{}).get('Status'),
                  restarts=state['RestartCount'], lines=lines)
    (OUT/(label+'-diagnostics.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def execute(label, filename, script, *args):
    path = OUT/(label+'-'+filename)
    if path.exists():
        raise RuntimeError('Refusing evidence overwrite')
    internal = '/tmp/'+path.name
    rc = subprocess.run(['docker','exec',CONTAINER,'python3','/tmp/h100-probes/'+script,
                         '--output',internal,*args],capture_output=True,text=True,timeout=1200)
    # Output consists of synthetic cases but store only a compact sanitized tail.
    (OUT/(path.name+'.log')).write_text(rc.stdout[-8000:])
    try:
        run('docker','cp',CONTAINER+':'+internal,str(path))
    except Exception:
        raise RuntimeError('Probe produced no evidence: '+filename) from None
    if rc.returncode:
        raise RuntimeError('Probe failed: '+filename)
    print(json.dumps({'completed':path.name}),flush=True)
    return path


def memory_observer(stop, path):
    with path.open('x') as out:
        while not stop.is_set():
            try:
                raw = run('nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
                          '--format=csv,noheader,nounits')
                row = {'time':time.time(),'gpu':raw.strip()}
            except Exception as exc:
                row = {'time':time.time(),'error':type(exc).__name__}
            out.write(json.dumps(row)+'\n')
            out.flush()
            stop.wait(5)


def main():
    assert socket.gethostname() == 'aquillm-dev2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--mode',choices=('screen','profile'),required=True)
    parser.add_argument('--profile',choices=('decode','prefill'),default='decode')
    parser.add_argument('--limits',default='1,2,4')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--requests',type=int,default=4)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+',args.prefix):
        raise ValueError('Invalid evidence prefix')
    limits = [int(x) for x in args.limits.split(',')]
    if not limits or any(x not in (1,2,4) for x in limits):
        raise ValueError('Unsupported sequence limit')
    lock = (OUT/'execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if list(OUT.glob(args.prefix+'-*')):
        raise RuntimeError('Evidence prefix already exists')
    initial = ready()
    protected = others()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if (STATE/'throughput.json').exists():
        switch('rollback')
        ready()
    else:
        switch('prepare')
    stopped = threading.Event()
    observer = threading.Thread(target=memory_observer,args=(stopped,OUT/(args.prefix+'-memory.jsonl')),daemon=True)
    observer.start()
    try:
        for limit in ([1] if args.mode=='profile' else limits):
            label = args.prefix+('-'+args.profile if args.mode=='profile' else '-seq'+str(limit))
            before = inspect()['Id']
            switch('switch',limit,args.profile if args.mode=='profile' else None)
            state = ready()
            if state['Id'] == before:
                run('docker','restart',CONTAINER)
                state = ready()
            if others() != protected:
                raise RuntimeError('Unrelated service identity drift')
            install_probes(args.mode)
            if args.mode == 'profile':
                execute(label,'request.json','profile_trace.py','--kind',args.profile)
                run('docker','cp',CONTAINER+':/tmp/aquillm-profile/'+args.profile,str(OUT/(label+'-traces')))
            else:
                execute(label,'quality.jsonl','throughput_bench.py','--mode','quality',
                        '--concurrency','4','--label',label)
                for workload in ('short','mixed'):
                    for concurrency in (1,4):
                        execute(label,workload+'-c'+str(concurrency)+'.jsonl','throughput_bench.py',
                                '--mode','performance','--workload',workload,'--concurrency',str(concurrency),
                                '--label',label,'--requests',str(args.requests),'--repeats',str(args.repeats))
            diagnostics(label)
            print(json.dumps({'arm_complete':label}),flush=True)
    except BaseException as exc:
        diagnostics(args.prefix+'-failure')
        (OUT/(args.prefix+'-failure.json')).write_text(json.dumps({'error':type(exc).__name__,
                                                               'reason':str(exc)})+'\n')
        raise
    finally:
        switch('rollback')
        restored = ready()
        result = {'image':restored['Image'],'healthy':True,'unrelated_services_unchanged':others()==protected,
                  'source_commit':run('git','-C',str(ROOT),'rev-parse','HEAD').strip()}
        (OUT/(args.prefix+'-restored.json')).write_text(json.dumps(result,indent=2)+'\n')
        stopped.set()
        observer.join(timeout=15)
        print(json.dumps(result),flush=True)
        if not result['unrelated_services_unchanged']:
            raise RuntimeError('Unrelated services changed during experiment')


if __name__ == '__main__':
    main()
