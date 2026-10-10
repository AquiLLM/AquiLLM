"""Reconcile only the owned test IDs proved deleted after a journal write failed."""
import argparse
import json
from pathlib import Path
import allocator_fixture_helper as helper

helper.setup(Path('/app'), Path('/tmp/chat_replay.py'))
from apps.chat.models import WSConversation
from apps.memory.models.jobs import ConversationMemoryJob

state_path = Path('/tmp/h100-allocator-fixture-state.json')
state = json.loads(state_path.read_text())
assert state['run_id'] == 'allocator20261010' and not state.get('principal_deleted')
user = helper.owner(state)
assert helper.fixture_proof(state)['ready']
args = argparse.Namespace(rows=[Path('/tmp/h100-allocator-chat-system-block1-owned.jsonl')], state=state_path)
rows = helper.rows_from(args, state)
ids = sorted(row['conversation_id'] for row in rows)
proof = json.loads(Path('/tmp/h100-allocator-chat-system-block1-proof.json').read_text())
assert proof['answer_proof_passed'] and len(ids) == 12
assert sorted(row['conversation_id'] for row in proof['conversations']) == ids
assert all(row['complete'] and row['label'] == 'system-block1' for row in rows)
assert all(all(row[field] for field in ('persisted_output_matches', 'exact_answer', 'model_matches', 'successful_finish'))
           for row in proof['conversations'])
assert not WSConversation.objects.filter(pk__in=ids).exists()
assert not WSConversation.objects.filter(owner=user).exists()
assert not ConversationMemoryJob.objects.filter(conversation_id__in=ids).exists()
assert state['cleared_conversation_ids'] == []
assert helper.profile_state(state)['profile_facts_empty']
state['cleared_conversation_ids'] = ids
helper.save(state_path, state)
print(json.dumps(dict(reconciled=True, run_id=state['run_id'], user_id=user.pk,
    cleared_conversation_ids=ids, proof_ids_match=True, all_recorded_ids_absent=True,
    no_other_owned_chats=True, owned_memory_jobs_absent=True, profile_facts_empty=True)))
