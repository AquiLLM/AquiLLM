"""Private, local-only frozen two-pair replay analysis. No network or mutations.

Usage: python chat_retry_analysis.py CAPTURE_DIR [--output PATH]
Writes descriptive JSON only; missing evidence is always explicit. Attempt1 is
not read. Visible answer latency is not raw model token TTFT. No CI/significance
calculation: two fixed-order system-then-mimalloc boot pairs are descriptive.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

METRICS = ('visible_ttft_seconds', 'final_seconds', 'persisted_seconds')
KINDS = ('chat', 'rag')
ARMS = ('system', 'mimalloc')
LABELS = tuple(f'{arm}-block{boot}' for boot in (1, 2) for arm in ARMS)
PREFIX = 'h100-allocator-chat-retry-'
PREFILL_IMAGE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
WEB_IMAGE = 'sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8'
PREFILL_FLAGS = dict(AQUILLM_H100_PREFILL='1', AQUILLM_H100_MTP_KERNEL='baseline',
                     AQUILLM_H100_SPLIT_POLICY='baseline', AQUILLM_H100_GDN='baseline')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def replay_source():
    source = next((candidate for ancestor in Path(__file__).resolve().parents
                   if (candidate := ancestor / 'scripts/h100_performance/chat_replay.py').is_file()), None)
    if source is None:
        raise FileNotFoundError('Cannot find repository scripts/h100_performance/chat_replay.py')
    return source


def oracle_functions():
    # Use the actual replay oracle/payload functions without importing network deps.
    source = replay_source()
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'normalized', 'exact_answer', 'payload_for'}
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name in names], type_ignores=[])
    namespace = {}
    exec(compile(selected, str(source), 'exec'), namespace)
    return namespace, hashlib.sha256(source.read_bytes()).hexdigest()


def p95(values):
    """Linear interpolated sample percentile; n=5 per boot/kind is descriptive."""
    values = sorted(values)
    position = .95 * (len(values) - 1)
    lo = math.floor(position)
    return values[lo] + (values[math.ceil(position)] - values[lo]) * (position - lo)


def summary(rows):
    return {metric: {'n': len(rows), 'median': statistics.median([r[metric] for r in rows]),
                     'p95': p95([r[metric] for r in rows])}
            for metric in METRICS} if rows else None


def process_checks(rows, arm, roles):
    errors = []
    if Counter(r.get('role') for r in rows) != Counter(roles):
        errors.append('process_role_coverage')
    for r in rows:
        role = r.get('role')
        libs = r.get('libraries')
        if r.get('configured_allocator') != arm:
            errors.append(f'{role}:configured_allocator')
        if arm == 'system':
            valid_libs = libs == []
        else:
            valid_libs = isinstance(libs, list) and bool(libs) and all(
                isinstance(s, str) and re.fullmatch(r'/opt/mimalloc/(?:[^/]+/)*libmimalloc\.so(?:\.\d+)*', s)
                and '..' not in s.split('/') for s in libs)
        if not valid_libs:
            errors.append(f'{role}:allocator_mappings')
        if role == 'api' and r.get('process_environment_reliable') is not True:
            errors.append('api:process_environment_reliable')
        # Only the engine can lose /proc environment visibility via setproctitle.
        allow_missing = role == 'engine' and r.get('process_environment_reliable') is False
        for key, expected in [('observed_env_allocator', arm),
                              ('observed_env_pythonmalloc', 'default')]:
            if r.get(key) != expected and not (allow_missing and r.get(key) is None):
                errors.append(f'{role}:{key}')
    return errors


def completion_checks(cleanup, marker, restoration):
    """Overall completion needs cleanup plus runner and independent restore proof.

    The independent verifier compares allocator_environment with captured original
    state and validates protected runtime/env. Its recorded assertion is required;
    missing environment keys here preserve absence rather than invent defaults.
    """
    errors = []
    for name, value in [('final_cleanup', cleanup), ('complete_marker', marker),
                        ('independent_restoration', restoration)]:
        if not isinstance(value, dict): errors.append(name + ':missing')
    if errors: return errors
    for key in ('user_and_fixture_deleted', 'live_mem0_vectors_empty', 'live_mem0_graph_nodes_empty'):
        if cleanup.get(key) is not True: errors.append('final_cleanup:' + key)
    for key in ('completed', 'cleanup_verified', 'both_services_healthy'):
        if marker.get(key) is not True: errors.append('complete_marker:' + key)
    if marker.get('restored_prefill_image') != PREFILL_IMAGE or marker.get('restored_web_image') != WEB_IMAGE:
        errors.append('complete_marker:restored_images')
    if restoration.get('host') != 'aquillm-dev2': errors.append('restoration:host')
    for key in ('protected_runtime_and_environment_validated', 'development_checkout_clean',
                'unrelated_services_unchanged_since_rollout'):
        if restoration.get(key) is not True: errors.append('restoration:' + key)
    for service, image in [('vllm', PREFILL_IMAGE), ('web', WEB_IMAGE)]:
        record = restoration.get(service, {})
        if record.get('image') != image or record.get('healthy') is not True:
            errors.append('restoration:' + service + ':image_or_health')
        public = record.get('allocator_environment')
        if not isinstance(public, dict) or any(k not in ('AQUILLM_ALLOCATOR', 'PYTHONMALLOC')
                or v != {'AQUILLM_ALLOCATOR': 'system', 'PYTHONMALLOC': 'default'}.get(k)
                for k, v in (public or {}).items()):
            errors.append('restoration:' + service + ':allocator_environment')
    if restoration.get('vllm', {}).get('h100_flags') != PREFILL_FLAGS:
        errors.append('restoration:h100_flags')
    model = restoration.get('model_response', {})
    if model.get('model') != 'qwen3.6:27b-mtp-awq' or model.get('exact_answer_passed') is not True or model.get('finish_reason') != 'stop':
        errors.append('restoration:model_response')
    if restoration.get('web_login_http_status') != 200:
        errors.append('restoration:web_login_http_status')
    try:
        recorded = datetime.fromisoformat(marker['at'])
        verified = datetime.fromisoformat(restoration['at'])
        if recorded.utcoffset() is None or verified.utcoffset() is None or verified < recorded:
            errors.append('restoration:timestamp_order')
    except (KeyError, ValueError, TypeError):
        errors.append('restoration:timestamp_order')
    return errors


def analyze(directory):
    directory = Path(directory)
    archive = replay_source().parents[2] / 'docs/audits/2026-10-10-h100-performance/mimalloc/application/retry'
    if directory.name != 'chat-retry' and directory.resolve() != archive.resolve():
        raise ValueError('Only the completed retry capture directory is allowed')
    funcs, oracle_hash = oracle_functions()
    missing, errors, manifest, blocks, all_rows = [], [], {}, {}, []

    def read(name, jsonl=False, exact_name=False):
        path = directory / (name if exact_name else PREFIX + name)
        if not path.exists():
            missing.append(path.name)
            return None
        raw = path.read_bytes()
        manifest[path.name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if jsonl else json.loads(raw)

    provision = read('provision.json')
    if not provision:
        return {'request_evidence_complete': False, 'evidence_complete': False, 'missing': missing}
    fixture = provision['fixture']
    expected_fixture = digest(fixture)
    expected_inputs = {kind: digest(funcs['payload_for'](kind, fixture)) for kind in KINDS}
    fixture_proof = provision.get('ingestion_proof')
    if not provision.get('provisioned') or not fixture_proof.get('ready'):
        errors.append('provision_not_ready')
    for label in LABELS:
        arm = label.split('-')[0]
        rows = read(label + '.jsonl', True)
        proof = read(label + '-proof.json')
        ready = read(label + '-ready.json')
        cleanup = read(label + '-cleanup.json')
        event_lines = read(label + '-events.json')
        process = {phase: (read(label + ('-after' if phase == 'after' else '') + '-processes.json'),
                           read(label + ('-after' if phase == 'after' else '') + '-web-processes.json'))
                   for phase in ('before', 'after')}
        if any(x is None for x in (rows, proof, ready, cleanup, event_lines)) or any(
                x is None for pair in process.values() for x in pair):
            continue
        local = []
        expected_ids = Counter((kind, repeat) for repeat in range(-1, 5) for kind in KINDS)
        if Counter((r.get('kind'), r.get('repeat')) for r in rows) != expected_ids:
            local.append('row_repeat_kind_coverage')
        if len({r.get('conversation_id') for r in rows}) != 12:
            local.append('conversation_uniqueness')
        checks = proof.get('conversations', [])
        if Counter(r.get('conversation_id') for r in checks) != Counter(r.get('conversation_id') for r in rows):
            local.append('persisted_proof_coverage')
        by_id = {r.get('conversation_id'): r for r in checks}
        valid = []
        for row in rows:
            bad = []
            kind = row.get('kind')
            if kind not in KINDS:
                bad.append('kind')
            else:
                if row.get('input_sha256') != expected_inputs[kind]: bad.append('input_hash')
                for key in ('output_text', 'stream_output_text'):
                    if not funcs['exact_answer'](kind, fixture, row.get(key)): bad.append(key)
            if row.get('fixture_sha256') != expected_fixture: bad.append('fixture_hash')
            if row.get('label') != label: bad.append('label')
            if row.get('warmup') is not (row.get('repeat', 0) < 0): bad.append('warmup')
            if row.get('oracle') != 'exact-chat-rag-v1': bad.append('oracle')
            if not row.get('complete') or row.get('error') is not None: bad.append('complete')
            if row.get('finish_reason') != 'stop': bad.append('finish')
            for key in ('stream_done', 'persisted_delta', 'action_completion_acknowledged'):
                if row.get(key) is not True: bad.append(key)
            if row.get('output_sha256') != hashlib.sha256(row.get('output_text', '').encode()).hexdigest():
                bad.append('output_hash')
            if row.get('expected_model') != fixture['expected_model']: bad.append('expected_model')
            if not all(isinstance(row.get(m), (int, float)) and not isinstance(row[m], bool)
                       and math.isfinite(row[m]) and row[m] >= 0 for m in METRICS):
                bad.append('latency')
            elif not row[METRICS[0]] <= row[METRICS[1]] <= row[METRICS[2]] <= row['total_seconds']:
                bad.append('latency_order')
            check = by_id.get(row.get('conversation_id'), {})
            if not all(check.get(k) is True for k in ('persisted_output_matches', 'exact_answer',
                                                      'model_matches', 'successful_finish')):
                bad.append('persisted_proof')
            if check.get('label') != label or check.get('kind') != kind or check.get('model') != fixture['expected_model']:
                bad.append('persisted_identity')
            if bad: local.append(f"row:{row.get('kind')}:{row.get('repeat')}:" + ','.join(bad))
            else: valid.append(row)
        if proof.get('answer_proof_passed') is not True: local.append('answer_proof')
        if not ready.get('ready_for_arm') or ready.get('fixture') != fixture_proof or proof.get('fixture') != fixture_proof:
            local.append('fixture_provenance_changed')
        if ready.get('profile', {}).get('profile_facts_empty') is not True or proof.get('profile', {}).get('profile_facts_empty') is not True:
            local.append('profile_not_empty')
        if Counter(cleanup.get('cleared_conversation_ids', [])) != Counter(r.get('conversation_id') for r in rows):
            local.append('cleanup_coverage')
        for key in ('live_mem0_vectors_empty', 'live_mem0_graph_nodes_empty', 'profile_facts_empty_after'):
            if cleanup.get(key) is not True: local.append('cleanup:' + key)
        if proof.get('configured_interface') != 'OpenAIInterface' or proof.get('configured_model') != fixture['expected_model'] or proof.get('configured_base_url') != 'http://vllm:8000/v1/':
            local.append('provider_configuration')
        mem = {}
        for phase, (model, web) in process.items():
            local += [phase + ':' + s for s in process_checks(model, arm, ('api', 'engine'))]
            local += [phase + ':' + s for s in process_checks(web, arm, ('web',))]
            mem[phase] = {r['role']: {'pid': r['pid'], 'memory': r.get('memory'), 'status': r.get('status')}
                          for r in model + web}
        if {role: v['pid'] for role, v in mem['before'].items()} != {role: v['pid'] for role, v in mem['after'].items()}:
            local.append('process_restart_during_arm')
        events = []
        for line in event_lines:
            try: events.append(line if isinstance(line, dict) else json.loads(line[line.index('{'):]))
            except (ValueError, TypeError): local.append('unparseable_event')
        llm = [e for e in events if e.get('event') == 'obs.llm.request_completed']
        rag = [e for e in events if e.get('event') == 'rag_direct_turn']
        route_counts = {'all_events': dict(Counter(e.get('event') for e in events)),
                        'llm_stage': dict(Counter(e.get('stage') for e in llm)),
                        'rag_status': {k: dict(Counter(str(e.get(k)) for e in rag)) for k in
                                       ('graph_status', 'retrieval_status', 'fixed_fallback_reason', 'proposed_score_status')},
                        'rag_graph_candidate_total': sum(e.get('graph_candidate_count', 0) for e in rag),
                        'raw_model_ttft_available': sum(isinstance(e.get('ttft_ms'), (int, float)) for e in llm)}
        route_counts['provider_token_work_including_warmups'] = {
            stage: {key: {'values': [e.get(key) for e in llm if e.get('stage') == stage],
                          'numeric_available_count': sum(type(e.get(key)) is int for e in llm if e.get('stage') == stage),
                          'numeric_sum': (sum(e[key] for e in llm if e.get('stage') == stage
                                             and type(e.get(key)) is int)
                                          if any(type(e.get(key)) is int for e in llm if e.get('stage') == stage) else None)}
                    for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens')}
            for stage in sorted({e.get('stage') for e in llm if isinstance(e.get('stage'), str)})}
        if len(llm) != 12 or len(rag) != 6: local.append('route_event_count')
        measured = [r for r in valid if not r['warmup']]
        blocks[label] = {'n_rows': len(rows), 'n_valid': len(valid), 'n_measured': len(measured),
                         'first_captured_at': min((r['captured_at'] for r in rows), default=None),
                         'last_captured_at': max((r['captured_at'] for r in rows), default=None),
                         'visible_equals_final_count': sum(r[METRICS[0]] == r[METRICS[1]] for r in valid),
                         'output_hash_counts': {k: dict(Counter(r['output_sha256'] for r in valid if r['kind'] == k)) for k in KINDS},
                         'latency': {k: summary([r for r in measured if r['kind'] == k]) for k in KINDS},
                         'events_including_warmups': route_counts, 'process_memory': mem,
                         'background': proof.get('background'), 'knowledge_graph': proof.get('knowledge_graph'),
                         'cleanup': cleanup, 'errors': local}
        errors += [label + ':' + s for s in local]
        all_rows += measured
    pooled = {arm: {kind: summary([r for r in all_rows if r['label'].startswith(arm + '-') and r['kind'] == kind])
                    for kind in KINDS} for arm in ARMS}
    if len({r['conversation_id'] for r in all_rows}) != len(all_rows):
        errors.append('cross_block_conversation_reuse')
    ordered = [blocks[label] for label in LABELS if label in blocks]
    if any(a['last_captured_at'] and b['first_captured_at'] and a['last_captured_at'] >= b['first_captured_at']
           for a, b in zip(ordered, ordered[1:])):
        errors.append('unexpected_fixed_boot_order')
    effects = {}
    for boot in (1, 2):
        s, m = blocks.get(f'system-block{boot}'), blocks.get(f'mimalloc-block{boot}')
        if s and m:
            effects[str(boot)] = {kind: {metric: {
                stat + '_latency_reduction_percent': 100 * (1 - m['latency'][kind][metric][stat] / s['latency'][kind][metric][stat])
                for stat in ('median', 'p95')} for metric in METRICS}
                                 for kind in KINDS if s['latency'][kind] and m['latency'][kind]}
    request_complete = not missing and not errors and len(blocks) == 4
    request_missing = missing.copy()
    cleanup = read('final-cleanup.json')
    marker = read('complete.json')
    restoration = read('h100-allocator-restoration-verified.json', exact_name=True)
    completion_errors = completion_checks(cleanup, marker, restoration)
    return {'request_evidence_complete': request_complete,
            'evidence_complete': request_complete and not completion_errors,
            'completion_errors': completion_errors, 'request_missing': request_missing,
            'final_cleanup': cleanup, 'complete_marker': marker, 'independent_restoration': restoration,
            'expected_total': 48, 'valid_total': sum(b['n_valid'] for b in blocks.values()),
            'measured_total': len(all_rows), 'warmups_excluded_expected': 8,
            'missing': missing, 'errors': errors, 'input_sha256': expected_inputs,
            'fixture_sha256': expected_fixture, 'replay_source_sha256': oracle_hash,
            'file_sha256': manifest, 'blocks': blocks, 'pooled_descriptive': pooled,
            'paired_boot_effects': effects,
            'limitations': ['Only retry captures read; aborted attempt excluded.',
                            'Both web and model services change allocator together; no component attribution.',
                            'Two fixed-order system-then-mimalloc boot pairs; no causal/significance/CI claim.',
                            'Five measured requests per kind/arm/boot; interpolated p95 is descriptive.',
                            'Visible answer TTFT is application first visible answer, not raw model TTFT.',
                            'Replay source SHA256 hashes exact bytes; CRLF/LF checkout conversion changes the hash without changing oracle behavior.',
                            'Provider token work includes warmups; identical visible answers do not prove identical hidden reasoning/model work.',
                            'Route log counts include warmups; no row-level correlation ID in replay output.',
                            'Provider configuration, stored model and aggregate logs support route evidence; no request-to-process dispatch proof.',
                            'Persisted answer and action ACK do not prove background memory/index/task completion.',
                            'Process memory endpoints/HWM are not a concurrent peak or soak/leak measurement.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.capture_directory)
    text = json.dumps(result, indent=2)
    if args.output: args.output.write_text(text + '\n', encoding='utf-8')
    else: print(text)
