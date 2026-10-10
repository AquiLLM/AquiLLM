"""Serialized development application replay after the allocator API experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
from run_prefill_blocks import run, wait_ready

assert socket.gethostname()=='aquillm-dev2'
assert json.loads(Path('/tmp/h100-allocator-api-complete.json').read_text())['completed']
C='compose-vllm-1'
WEB='compose-web-1'
ROOT=Path('/home/exouser/AquiLLM-h100')
SWITCH=str(ROOT/'scripts/h100_performance/allocator_switch.py')
WEB_SWITCH=str(ROOT/'scripts/h100_performance/web_allocator_switch.py')
IMAGE='sha256:dc14ba6ec72907fdcbc097a08eb99d69f104d9817a6e5758d29b5c819694926c'
PREFILL='sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
WEB_BASE='sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8'
WEB_IMAGE=json.loads(Path('/tmp/h100-web-mimalloc-build.json').read_text())['image']
RUN_ID='allocator20261010r2'
STATE='/tmp/h100-allocator-fixture-retry-state.json'
FIXTURE='/tmp/h100-allocator-fixture-retry.json'
HELPER=['docker','exec','-i',WEB,'/opt/venv/bin/python','/tmp/allocator_fixture_helper_retry.py']
COMMON=['--repo-root','/app','--replay-script','/tmp/chat_replay.py','--state',STATE]
rows_files=[]
provision_attempted=False
measurements_complete=False
cleanup_verified=False
web_restored=False
model_restored=False

def others():
    return {line.split()[0]:line.split()[1] for line in run('docker','ps','--format','{{.Names}} {{.ID}}').splitlines()
            if line.split()[0] not in (C,WEB)}

def wait_web(image):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        current=json.loads(run('docker','inspect',WEB))[0]
        assert current['Image']==image and current['State']['Running']
        if current['State'].get('Health',{}).get('Status')=='healthy':
            return
        time.sleep(2)
    raise RuntimeError('web_readiness_timeout')

def install_web_files():
    run('docker','cp',str(ROOT/'scripts/h100_performance/chat_replay.py'),WEB+':/tmp/chat_replay.py')
    run('docker','cp','/tmp/allocator_fixture_helper_retry.py',WEB+':/tmp/allocator_fixture_helper_retry.py')
    for path in (STATE,FIXTURE):
        if Path(path).exists():
            run('docker','cp',path,WEB+':'+path)
    # docker cp defaults to root ownership inside a replacement container.
    owned_paths=[path for path in (STATE,FIXTURE) if Path(path).exists()]
    if owned_paths:
        uid=run('docker','exec',WEB,'id','-u').strip()
        gid=run('docker','exec',WEB,'id','-g').strip()
        assert uid.isdecimal() and gid.isdecimal()
        run('docker','exec','--user','0',WEB,'chown',uid+':'+gid,*owned_paths)
        run('docker','exec',WEB,'/opt/venv/bin/python','-c',
            'import sys; [open(path, "r+").close() for path in sys.argv[1:]]',*owned_paths)

def helper(mode,*args,credentials=None,output=None):
    result=subprocess.run(HELPER+[mode]+COMMON+list(args),text=True,
                          input=json.dumps(credentials)+'\n' if credentials else '',
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # Helper has a fixed, credential-free output contract. Never expose stderr.
    lines=[line for line in result.stdout.splitlines() if line.startswith('{')]
    assert lines, 'fixture_helper_no_structured_result'
    value=json.loads(lines[-1])
    exists=subprocess.run(['docker','exec',WEB,'test','-f',STATE],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if exists.returncode==0:
        run('docker','cp',WEB+':'+STATE,STATE)
    if output:
        Path(output).write_text(json.dumps(value,indent=2)+'\n')
    if result.returncode:
        print(json.dumps({'fixture_helper_failure':mode,'diagnostic':value}),flush=True)
        raise RuntimeError('fixture_helper_'+mode+'_'+str(value.get('error_type','failed')))
    return value

def capture(label,allocator):
    run('docker','cp','/tmp/probe_serving_allocator.py',C+':/tmp/probe_serving_allocator.py')
    value=json.loads(run('docker','exec',C,'python3','/tmp/probe_serving_allocator.py',allocator))
    Path('/tmp/h100-allocator-chat-retry-'+label+'-processes.json').write_text(json.dumps(value,indent=2)+'\n')
    run('docker','cp','/tmp/probe_web_allocator.py',WEB+':/tmp/probe_web_allocator.py')
    web=json.loads(run('docker','exec',WEB,'/opt/venv/bin/python','/tmp/probe_web_allocator.py',allocator))
    Path('/tmp/h100-allocator-chat-retry-'+label+'-web-processes.json').write_text(json.dumps(web,indent=2)+'\n')

def replay(label,credentials):
    path='/tmp/h100-allocator-chat-retry-'+label+'.jsonl'
    assert not Path(path).exists()
    started=datetime.now(timezone.utc).isoformat()
    with Path(path+'.log').open('w') as log:
        result=subprocess.run(['docker','exec','-i',WEB,'/opt/venv/bin/python',
            '/tmp/chat_replay.py','--base-url','http://127.0.0.1:8080',
            '--fixture',FIXTURE,'--label',label,'--output',path,'--repeats','5','--warmup','1',
            '--timeout','180','--credentials-stdin','--keep-conversations'],
            input=json.dumps(credentials)+'\n',text=True,stdout=log,stderr=subprocess.STDOUT)
    run('docker','cp',WEB+':'+path,path)
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    owned=[row for row in rows if type(row.get('conversation_id')) is int]
    if owned:
        owned_path=path.replace('.jsonl','-owned.jsonl')
        Path(owned_path).write_text(''.join(json.dumps(row)+'\n' for row in owned))
        rows_files.append(owned_path)
    # Keep only known aggregate provider/RAG events, never general application logs.
    logs=run('docker','logs','--since',started,'--timestamps',WEB)
    events=[line for line in logs.splitlines() if any(name in line for name in
            ('obs.llm.request_completed','rag_direct_turn','obs.rag.preservation_turn'))]
    Path(path.replace('.jsonl','-events.json')).write_text(json.dumps(events,indent=2)+'\n')
    assert result.returncode==0 and len(rows)==12 and all(row['complete'] for row in rows)
    proof=helper('prove','--rows',path,'--wait-seconds','0',output=path.replace('.jsonl','-proof.json'))
    assert proof['answer_proof_passed'] and proof['fixture']['ready']
    assert proof['configured_model']=='qwen3.6:27b-mtp-awq'
    assert proof['configured_base_url']=='http://vllm:8000/v1/'
    helper('clear_conversations','--rows',path,output=path.replace('.jsonl','-cleanup.json'))
    print(json.dumps({'completed':label,'requests':len(rows),'answer_proof_passed':True}),flush=True)

before=others()
credentials={'username':'allocator-replay-'+RUN_ID,'password':secrets.token_urlsafe(36)}
try:
    wait_ready(PREFILL)
    wait_web(WEB_BASE)
    assert not Path(STATE).exists() and not Path(FIXTURE).exists()
    install_web_files()
    provision_attempted=True
    provision=helper('provision','--run-id',RUN_ID,'--fixture',FIXTURE,'--model','qwen3.6:27b-mtp-awq',
        '--base-url','http://127.0.0.1:8080','--wait-seconds','300',credentials=credentials,
        output='/tmp/h100-allocator-chat-retry-provision.json')
    assert provision['provisioned']
    run('docker','cp',WEB+':'+FIXTURE,FIXTURE)
    print(json.dumps({'fixture_ready':True,'run_id':RUN_ID}),flush=True)
    for block in (1,2):
        for allocator in ('system','mimalloc'):
            label=f'{allocator}-block{block}'
            print(run('python3',SWITCH,'switch','--allocator',allocator),flush=True)
            wait_ready(IMAGE)
            print(run('python3',WEB_SWITCH,'switch','--allocator',allocator),flush=True)
            wait_web(WEB_IMAGE)
            install_web_files()
            assert others()==before
            capture(label,allocator)
            helper('check_ready',output='/tmp/h100-allocator-chat-retry-'+label+'-ready.json')
            replay(label,credentials)
            capture(label+'-after',allocator)
            assert others()==before
    measurements_complete=True
finally:
    try:
        if provision_attempted:
            install_web_files()
            for path in rows_files:
                run('docker','cp',path,WEB+':'+path)
            cleanup=helper('final_cleanup','--rows',*rows_files,output='/tmp/h100-allocator-chat-retry-final-cleanup.json')
            cleanup_verified=bool(cleanup['user_and_fixture_deleted'] and cleanup['live_mem0_vectors_empty'] and cleanup['live_mem0_graph_nodes_empty'])
            assert cleanup_verified
    finally:
        try:
            print(run('python3',WEB_SWITCH,'rollback'),flush=True)
            wait_web(WEB_BASE)
            web_restored=True
        finally:
            print(run('python3',SWITCH,'rollback'),flush=True)
            wait_ready(PREFILL)
            model_restored=True
            assert others()==before
            print(json.dumps({'prefill_restored_healthy':model_restored,'web_restored_healthy':web_restored}),flush=True)

assert measurements_complete and cleanup_verified and web_restored and model_restored
Path('/tmp/h100-allocator-chat-retry-complete.json').write_text(json.dumps(dict(completed=True,
    cleanup_verified=True,restored_prefill_image=PREFILL,restored_web_image=WEB_BASE,
    both_services_healthy=True,at=datetime.now(timezone.utc).isoformat()))+'\n')
