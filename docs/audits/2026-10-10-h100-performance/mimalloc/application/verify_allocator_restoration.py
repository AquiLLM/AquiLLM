"""Independent read-only verification after the completed allocator replay."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys

assert socket.gethostname() == 'aquillm-dev2'

def run(*args, input=None):
    return subprocess.check_output(args, input=input, text=True)

marker = json.loads(Path('/tmp/h100-allocator-chat-retry-complete.json').read_text())
assert marker['completed'] and marker['cleanup_verified'] and marker['both_services_healthy']
directory = Path.home()/'.config/aquillm/h100-performance'
sys.path.insert(0, '/home/exouser/AquiLLM-h100/scripts/h100_performance')
import allocator_switch
import web_allocator_switch
expected_flags = dict(AQUILLM_H100_PREFILL='1', AQUILLM_H100_MTP_KERNEL='baseline',
                      AQUILLM_H100_SPLIT_POLICY='baseline', AQUILLM_H100_GDN='baseline')
records = {}
containers = {}
states = {}
for service, state_name, image_key in [('vllm', 'allocator.json', 'restored_prefill_image'),
                                     ('web', 'web-allocator.json', 'restored_web_image')]:
    state = json.loads((directory/state_name).read_text())
    container = json.loads(run('docker', 'inspect', 'compose-'+service+'-1'))[0]
    env = allocator_switch.base.environment(container)
    public = allocator_switch.public_environment(env)
    allocator_switch.reject_preloads(env)
    assert public == state['baseline_allocator_environment']
    assert container['Image'] == marker[image_key]
    assert container['State']['Running'] and container['State']['Health']['Status'] == 'healthy'
    flags = {key: env.get(key) for key in expected_flags} if service == 'vllm' else None
    if flags is not None:
        assert flags == expected_flags
    records[service] = dict(image=container['Image'], healthy=True,
                            allocator_environment=public, h100_flags=flags)
    containers[service] = container
    states[service] = state

original = json.loads((directory/'baseline.json').read_text())
allocator_switch.validate_configuration(states['vllm'], original, containers['vllm'])
web_allocator_switch.validate_configuration(states['web'], containers['web'], containers['vllm'])
records['protected_runtime_and_environment_validated'] = True

model_probe = '''import json,os,urllib.request
body=dict(model="qwen3.6:27b-mtp-awq",messages=[dict(role="user",content="Reply with exactly OK and nothing else.")],temperature=0,max_tokens=32,chat_template_kwargs=dict(enable_thinking=False))
headers={"Content-Type":"application/json"}
if os.environ.get("VLLM_API_KEY"): headers["Authorization"]="Bearer "+os.environ["VLLM_API_KEY"]
req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",data=json.dumps(body).encode(),headers=headers)
response=json.load(urllib.request.urlopen(req,timeout=60))
choice=response["choices"][0]
assert choice["message"]["content"].strip()=="OK" and choice["finish_reason"]=="stop"
print(json.dumps(dict(model=response["model"],exact_answer_passed=True,finish_reason=choice["finish_reason"])))
'''
records['model_response'] = json.loads(run('docker', 'exec', '-i', 'compose-vllm-1', 'python3', '-', input=model_probe))
web_probe = 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8080/accounts/login/",timeout=30).status)'
assert run('docker', 'exec', 'compose-web-1', '/opt/venv/bin/python', '-c', web_probe).strip() == '200'
records['web_login_http_status'] = 200
provision = json.loads(Path('/tmp/h100-allocator-chat-retry-provision.json').read_text())
fixture = provision['fixture']
assert fixture['run_id'] == 'allocator20261010r2'
conversation_ids = []
for boot in (1, 2):
    for arm in ('system', 'mimalloc'):
        rows = [json.loads(line) for line in Path(f'/tmp/h100-allocator-chat-retry-{arm}-block{boot}.jsonl').read_text().splitlines()]
        conversation_ids.extend(row['conversation_id'] for row in rows)
assert len(conversation_ids) == len(set(conversation_ids)) == 48
scope = dict(user_id=provision['user_id'], collection_id=fixture['collection_id'],
             document_id=fixture['document_id'], conversation_ids=conversation_ids)
cleanup_probe = '''import contextlib,io,json,logging,os,sys
scope=json.loads(sys.argv[1])
logging.disable(logging.CRITICAL)
os.environ.setdefault("DJANGO_SETTINGS_MODULE","aquillm.settings")
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 import django
 django.setup()
 from django.contrib.auth import get_user_model
 from django.db import connection
 from apps.chat.models import WSConversation
 from apps.collections.models import Collection
 from apps.documents.models import VTTDocument,TextChunk
 with connection.cursor() as cursor: cursor.execute("SET statement_timeout = '5s'")
 result=dict(user_absent=not get_user_model().objects.filter(pk=scope['user_id']).exists(),collection_absent=not Collection.objects.filter(pk=scope['collection_id']).exists(),document_absent=not VTTDocument.objects.filter(id=scope['document_id']).exists(),chunks_absent=not TextChunk.objects.filter(doc_id=scope['document_id']).exists(),conversations_absent=not WSConversation.objects.filter(pk__in=scope['conversation_ids']).exists(),other_owned_conversations_absent=not WSConversation.objects.filter(owner_id=scope['user_id']).exists())
assert all(result.values())
print(json.dumps(result))
'''
records['independent_fixture_cleanup'] = json.loads(run('docker', 'exec', 'compose-web-1',
    '/opt/venv/bin/python', '-c', cleanup_probe, json.dumps(scope)))
root = '/home/exouser/AquiLLM'
records['development_commit'] = run('git', '-C', root, 'rev-parse', 'HEAD').strip()
assert run('git', '-C', root, 'branch', '--show-current').strip() == 'development'
assert not run('git', '-C', root, 'status', '--porcelain').strip()
records['development_checkout_clean'] = True
records['host'] = socket.gethostname()
rollout = json.loads(Path('/tmp/h100-development-rollout.json').read_text())
excluded = {'compose-vllm-1', 'compose-web-1'}
before = {line.split()[0]: line.split()[1] for line in rollout['containers'] if line.split()[0] not in excluded}
after = {line.split()[0]: line.split()[1] for line in run('docker', 'ps', '--format', '{{.Names}} {{.ID}}').splitlines() if line.split()[0] not in excluded}
records['unrelated_services_unchanged_since_rollout'] = before == after
records['unrelated_service_ids'] = after
records['at'] = datetime.now(timezone.utc).isoformat()
Path('/tmp/h100-allocator-restoration-verified.json').write_text(json.dumps(records, indent=2)+'\n')
print(json.dumps(records, indent=2))
