"""Build a plugin-only immutable candidate without touching serving containers."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path('/home/exouser/AquiLLM-throughput')
OUT = Path('/tmp/h100-throughput')
BASE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE).strip()


def main():
    assert socket.gethostname() == 'aquillm-dev2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.commit)
    assert run('git','-C',str(ROOT),'rev-parse','HEAD') == args.commit
    assert not run('git','-C',str(ROOT),'status','--porcelain')
    lock = (OUT/'execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = 'candidate-'+args.commit[:12]
    result_path = OUT/(name+'.json')
    assert not result_path.exists()
    image_path = OUT/(name+'.image-id')
    tag = 'aquillm-h100-throughput:'+args.commit[:12]
    command = ['docker','build','--pull=false','--build-arg','BASE_IMAGE='+BASE,
               '--iidfile',str(image_path),'-t',tag,'-f',
               str(ROOT/'deploy/docker/vllm/Dockerfile.throughput-baseline'),str(ROOT)]
    with (OUT/(name+'-build.log')).open('x') as log:
        proc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    if proc.returncode:
        raise RuntimeError('Candidate build failed; inspect bounded build log')
    candidate = image_path.read_text().strip()
    assert re.fullmatch('sha256:[0-9a-f]{64}',candidate)
    old = json.loads(run('docker','image','inspect',BASE))[0]
    new = json.loads(run('docker','image','inspect',candidate))[0]
    assert old['Config'] == new['Config'], 'Plugin build changed runtime image configuration'
    created = run('docker','create',candidate,'/bin/true')
    try:
        for phase in ('before','after'):
            run('docker','cp',created+':/opt/throughput-baseline-evidence/'+phase+'.json',
                str(OUT/(name+'-'+phase+'.json')))
    finally:
        run('docker','rm',created)
    before = json.loads((OUT/(name+'-before.json')).read_text())
    after = json.loads((OUT/(name+'-after.json')).read_text())
    assert before == after
    result = dict(source_commit=args.commit,baseline=BASE,candidate=candidate,
                  image_configuration_unchanged=True,packages_checked=len(after['packages']),
                  plugins=after['plugins'],genesis=after['genesis'],build_command=command)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
