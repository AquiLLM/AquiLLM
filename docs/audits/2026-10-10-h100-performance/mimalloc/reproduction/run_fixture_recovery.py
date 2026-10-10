"""Coordinator for the explicitly identified failed test-journal write."""
import json
from pathlib import Path
import socket
import subprocess
from run_prefill_blocks import run, wait_ready

assert socket.gethostname() == 'aquillm-dev2'
assert '"prefill_restored_healthy": true, "web_restored_healthy": true' in Path('/tmp/h100-allocator-chat-run.log').read_text()
wait_ready('sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7')
web = 'compose-web-1'
current=json.loads(run('docker','inspect',web))[0]
assert current['Image']=='sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8'
assert current['State']['Running'] and current['State']['Health']['Status']=='healthy'
files = ['allocator_fixture_helper.py', 'recover_allocator_fixture.py',
    'h100-allocator-fixture-state.json', 'h100-allocator-fixture.json',
    'h100-allocator-chat-system-block1-owned.jsonl', 'h100-allocator-chat-system-block1-proof.json']
for name in files:
    run('docker','cp','/tmp/'+name,web+':/tmp/'+name)
run('docker','cp','/home/exouser/AquiLLM-h100/scripts/h100_performance/chat_replay.py',web+':/tmp/chat_replay.py')
uid=run('docker','exec',web,'id','-u').strip()
gid=run('docker','exec',web,'id','-g').strip()
assert uid.isdecimal() and gid.isdecimal()
run('docker','exec','--user','0',web,'chown',uid+':'+gid,'/tmp/h100-allocator-fixture-state.json')
for script, args, output in [
    ('recover_allocator_fixture.py', [], '/tmp/h100-allocator-chat-recovery.json'),
    ('allocator_fixture_helper.py', ['final_cleanup','--repo-root','/app','--replay-script','/tmp/chat_replay.py',
        '--state','/tmp/h100-allocator-fixture-state.json','--rows','/tmp/h100-allocator-chat-system-block1-owned.jsonl'],
        '/tmp/h100-allocator-chat-recovery-cleanup.json')]:
    result=subprocess.run(['docker','exec',web,'/opt/venv/bin/python','/tmp/'+script]+args,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    lines=[line for line in result.stdout.splitlines() if line.startswith('{')]
    assert lines, 'no_structured_recovery_result'
    value=json.loads(lines[-1])
    Path(output).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)
    run('docker','cp',web+':/tmp/h100-allocator-fixture-state.json','/tmp/h100-allocator-fixture-state.json')
    assert result.returncode == 0, 'scoped_recovery_failed'
run('docker','cp',web+':/tmp/h100-allocator-fixture-state.json','/tmp/h100-allocator-fixture-state.json')
