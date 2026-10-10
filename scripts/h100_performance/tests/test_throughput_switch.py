"""Only bounded sequence/profiler choices may vary; recovery retains exact text."""
import copy
import importlib
import json
import os
from pathlib import Path
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dev_switch as base

IMAGE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
FLAGS = dict(AQUILLM_H100_PREFILL='1', AQUILLM_H100_MTP_KERNEL='baseline',
             AQUILLM_H100_SPLIT_POLICY='baseline', AQUILLM_H100_GDN='baseline')
ARGS = "  --max-num-seqs=1 --max-num-batched-tokens 4096 --speculative-config " \
       "'{\"method\":\"mtp\",\"num_speculative_tokens\":4}' --kv-cache-dtype turboquant_k8v4  "


def helper():
    try:
        return importlib.import_module('throughput_switch')
    except ModuleNotFoundError:
        pytest.fail('throughput switch helper is missing')


def envset(current, values):
    current['Config']['Env'] = [f'{key}={value}' for key, value in values.items()]


@pytest.fixture
def baseline():
    values = dict(PATH='/usr/bin', TOKEN='synthetic-secret', MODEL='same-model',
                  VLLM_PLUGINS='sndr,existing-plugin', VLLM_EXTRA_ARGS=ARGS, **FLAGS)
    current = dict(Name='/compose-vllm-1', Image=IMAGE, State=dict(Running=True),
        Config=dict(Env=[], Cmd=['serve'], Entrypoint=['/genesis_entrypoint.sh'], User='',
            Labels={'com.docker.compose.project': 'compose', 'com.docker.compose.service': 'vllm',
                    'com.docker.compose.project.working_dir': '/repo',
                    'com.docker.compose.project.config_files': 'compose.yml',
                    'com.docker.compose.config-hash': 'verified'}),
        HostConfig=dict(Binds=['/cache:/cache'], LogConfig={'Type': 'json-file'}),
        Mounts=[dict(Source='/cache', Destination='/cache', RW=True)])
    envset(current, values)
    resolved = dict(image=IMAGE, environment=values.copy(), command=['serve'],
                    volumes=[dict(source='/cache', target='/cache')])
    original = base.prepare_state(dict(image=IMAGE, project='compose', working_dir='/repo',
                                      files=['compose.yml']), current, resolved, 'verified', verified=True)
    image = dict(Id=IMAGE, Config=dict(Env=['PATH=/usr/bin'], Cmd=['serve'],
                                      Entrypoint=['/genesis_entrypoint.sh']))
    return original, current, resolved, image


def prepared(baseline):
    return helper().prepare_state(*baseline[:3], 'verified', baseline[3])


@pytest.mark.parametrize('text', [ARGS, "--max-num-seqs 1 --seed 42 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"])
def test_transform_changes_only_sequence_value_semantically(text):
    tokens = shlex.split(helper().transform_args(text, max_num_seqs=4))
    before = shlex.split(text)
    if '--max-num-seqs=1' in before:
        before[before.index('--max-num-seqs=1')] = '--max-num-seqs=4'
    else:
        before[before.index('--max-num-seqs') + 1] = '4'
    assert tokens == before


@pytest.mark.parametrize('limit', [0, 3, 8, True, '4'])
def test_transform_rejects_unbounded_sequence_limits(limit):
    with pytest.raises(SystemExit): helper().transform_args(ARGS, max_num_seqs=limit)


@pytest.mark.parametrize('text', [
    '--max-num-seqs 1 --max-num-seqs=2', '--max-num-seqs', '--max-num-seqs --seed 7',
    '--max-num-seqs nope', '--seed 1 --seed 2', '--max_num_seqs 1',
    '--profiler-config nope', '--profiler-config {} --profiler-config {}',
    "--profiler-config '{\"profiler\":\"torch\",\"profiler\":\"torch\"}'", '--seed "unterminated',
])
def test_transform_rejects_ambiguous_or_malformed_flags(text):
    with pytest.raises(SystemExit): helper().transform_args(text, max_num_seqs=2)


@pytest.mark.parametrize('profile,counts', [('decode', (2, 2, 32, 32)), ('prefill', (0, 0, 1, 1))])
def test_profiler_is_bounded_and_preserves_other_arguments(profile, counts):
    tokens = shlex.split(helper().transform_args('--max-num-seqs 1 --seed 7', max_num_seqs=1, profile=profile))
    assert tokens[:4] == ['--max-num-seqs', '1', '--seed', '7']
    assert tokens[4] == '--profiler-config'
    config = json.loads(tokens[5])
    assert config == dict(profiler='torch', torch_profiler_dir='/tmp/aquillm-profile/' + profile,
        ignore_frontend=True, torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        torch_profiler_dump_cuda_time_total=False, wait_iterations=counts[0], warmup_iterations=counts[1],
        active_iterations=counts[2], max_iterations=counts[3])
    with pytest.raises(SystemExit): helper().transform_args(ARGS, max_num_seqs=2, profile=profile)


def test_existing_profiler_cannot_smuggle_paths_or_options():
    for config in [dict(profiler='torch', torch_profiler_dir='/etc'),
                   dict(profiler='torch', active_iterations=100000), dict(profiler='cuda'),
                   dict(profiler='torch', torch_profiler_with_stack=True)]:
        text = '--max-num-seqs 1 --profiler-config ' + shlex.quote(json.dumps(config))
        with pytest.raises(SystemExit): helper().transform_args(text, max_num_seqs=1, profile='decode')


def test_profiler_rejects_json_types_that_only_compare_equal_to_bounds():
    config = dict(profiler='torch', torch_profiler_dir='/tmp/aquillm-profile/prefill',
        ignore_frontend=True, torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        torch_profiler_dump_cuda_time_total=False, wait_iterations=0, warmup_iterations=0,
        active_iterations=True, max_iterations=1)
    with pytest.raises(SystemExit):
        helper().transform_args('--profiler-config ' + shlex.quote(json.dumps(config)), max_num_seqs=1)


def test_prepare_binds_authoritative_state_and_only_two_plaintext_values(baseline):
    original = copy.deepcopy(baseline[0])
    state = prepared(baseline)
    assert state['image'] == IMAGE
    assert state['original_state_digest'] == base.digest(original)
    assert state['baseline_environment'] == {'VLLM_EXTRA_ARGS': ARGS}
    assert 'synthetic-secret' not in json.dumps(state)
    assert baseline[0] == original


@pytest.mark.parametrize('change', ['image', 'original-image', 'hash', 'secret', 'plugin', 'mtp', 'command', 'mount', 'resolved'])
def test_prepare_rejects_unverified_running_baseline(baseline, change):
    original, current, resolved, image = copy.deepcopy(baseline)
    values = base.environment(current)
    if change == 'image': current['Image'] = 'sha256:' + 'a' * 64
    elif change == 'original-image': original['image'] = 'sha256:' + 'a' * 64
    elif change == 'secret': values['TOKEN'] = 'changed'
    elif change == 'plugin': values['VLLM_PLUGINS'] = 'sndr'
    elif change == 'mtp': values['AQUILLM_H100_MTP_KERNEL'] = 'fused'
    elif change == 'command': current['Config']['Cmd'] = ['other']
    elif change == 'mount': current['Mounts'][0]['RW'] = False
    elif change == 'resolved': resolved['environment']['MODEL'] = 'other'
    envset(current, values)
    with pytest.raises(SystemExit):
        helper().prepare_state(original, current, resolved, 'wrong' if change == 'hash' else 'verified', image)


@pytest.mark.parametrize('args_present,scopes', [(True, None), (False, None), (True, ''), (True, '0'), (True, '1')])
def test_rollback_restores_exact_text_and_presence_even_when_candidate_exited(baseline, args_present, scopes):
    original, current, resolved, image = baseline
    values = base.environment(current)
    if not args_present: values.pop('VLLM_EXTRA_ARGS')
    if scopes is not None: values['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] = scopes
    envset(current, values)
    resolved['environment'] = values.copy()
    original['environment_digests'] = base.protected_environment(values)
    state = prepared(baseline)
    candidate, _ = helper().make_override(state, original, current, profile='decode', process_env={})
    selected = candidate['services']['vllm']['environment']
    values.update({key: selected[key] for key in ('VLLM_EXTRA_ARGS', 'VLLM_CUSTOM_SCOPES_FOR_PROFILING')})
    envset(current, values)
    current['State'] = dict(Running=False, ExitCode=1)
    rollback, process = helper().make_override(state, original, current, rollback=True, process_env={})
    environment = rollback['services']['vllm']['environment']
    assert environment['VLLM_EXTRA_ARGS'] == (ARGS if args_present else None)
    assert environment['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] == scopes
    assert environment['TOKEN'] == '${AQUILLM_H100_CAPTURED_ENV_TOKEN}'
    assert process['AQUILLM_H100_CAPTURED_ENV_TOKEN'] == 'synthetic-secret'
    assert 'synthetic-secret' not in json.dumps(rollback)


@pytest.mark.parametrize('change', ['image', 'extra-arg', 'profile', 'scopes', 'plugin', 'secret', 'mtp', 'command', 'mount', 'original'])
def test_candidate_drift_blocks_both_switch_and_rollback(baseline, change):
    state = prepared(baseline)
    original, current, _, _ = copy.deepcopy(baseline)
    values = base.environment(current)
    values['VLLM_EXTRA_ARGS'] = helper().transform_args(ARGS, max_num_seqs=4)
    if change == 'image': current['Image'] = 'sha256:' + 'a' * 64
    elif change == 'extra-arg': values['VLLM_EXTRA_ARGS'] += ' --dtype bfloat16'
    elif change == 'profile': values['VLLM_EXTRA_ARGS'] += " --profiler-config '{}'"
    elif change == 'scopes': values['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] = '1'
    elif change == 'plugin': values['VLLM_PLUGINS'] = 'sndr'
    elif change == 'secret': values['TOKEN'] = 'changed'
    elif change == 'mtp': values['AQUILLM_H100_MTP_KERNEL'] = 'fused'
    elif change == 'command': current['Config']['Cmd'] = ['changed']
    elif change == 'mount': current['HostConfig']['Binds'] = ['/other:/cache']
    elif change == 'original': original['service_digest'] = 'changed'
    envset(current, values)
    for rollback in (False, True):
        with pytest.raises(SystemExit): helper().make_override(state, original, current, rollback=rollback, process_env={})


def test_cli_missing_recovery_and_host_guard_precede_docker(tmp_path, monkeypatch):
    module = helper()
    monkeypatch.setattr(module.socket, 'gethostname', lambda: 'production')
    monkeypatch.setattr(module.base, 'run', lambda *a, **k: pytest.fail('preflight touched Docker'))
    with pytest.raises(SystemExit): module.main(['switch', '--state-dir', str(tmp_path), '--max-num-seqs', '2'])
    monkeypatch.setattr(module.socket, 'gethostname', lambda: 'aquillm-dev2')
    monkeypatch.setattr(module, 'ORIGINAL_STATE', tmp_path / 'missing-original.json')
    with pytest.raises(SystemExit): module.main(['rollback', '--state-dir', str(tmp_path)])


def cli_runtime(baseline, tmp_path, monkeypatch, *, prepare=False, post_drift=None, up_failure=False):
    """Intercept Docker/Compose only; execute state, overrides and validators."""
    module = helper()
    original, current, service, image = copy.deepcopy(baseline)
    original_path = tmp_path / 'authoritative.json'
    original_path.write_text(json.dumps(original))
    directory = tmp_path / 'private'
    directory.mkdir()
    if not prepare:
        module.write_private(directory / 'throughput.json', prepared(baseline), exclusive=True)
    runtime = dict(current=current, calls=[], fail=up_failure, images={IMAGE: image})
    monkeypatch.setattr(module, 'ORIGINAL_STATE', original_path)
    monkeypatch.setattr(module.socket, 'gethostname', lambda: 'aquillm-dev2')
    monkeypatch.setattr(module.os, 'environ', {'SYNTHETIC_HOST': '1'})

    def run(command, env=None, input=None):
        runtime['calls'].append(command)
        if command[:2] == ['docker', 'inspect']:
            assert command == ['docker', 'inspect', 'compose-vllm-1']
            return json.dumps([runtime['current']])
        if command[:3] == ['docker', 'image', 'inspect']:
            return json.dumps([runtime['images'][command[-1]]])
        if command[-3:] == ['config', '--format', 'json']:
            resolved = copy.deepcopy(service)
            override_path = directory / 'throughput-next.json'
            if str(override_path) in command:
                override = json.loads(override_path.read_text())['services']['vllm']
                resolved['image'] = override['image']
                for name, value in override['environment'].items():
                    if value is None:
                        resolved['environment'].pop(name, None)
                    elif value.startswith('${AQUILLM_H100_CAPTURED_ENV_') and value.endswith('}'):
                        resolved['environment'][name] = env[value[2:-1]]
                    else:
                        resolved['environment'][name] = value.replace('$$', '$')
            if post_drift == 'resolved': resolved['environment']['TOKEN'] = 'changed'
            runtime['resolved'] = resolved
            # Model Compose's documented reusable-dollar escaping.
            return json.dumps({'services': {'vllm': resolved}}).replace('$', '$$')
        if command[-3:] == ['config', '--hash', 'vllm']:
            assert input is not None
            changed = runtime['resolved'] != service
            return 'vllm ' + ('after' if changed else runtime['current']['Config']['Labels']['com.docker.compose.config-hash']) + '\n'
        assert command[-5:] == ['up', '-d', '--no-deps', '--no-build', 'vllm']
        restored = copy.deepcopy(current)
        restored['Image'] = runtime['resolved']['image']
        envset(restored, runtime['resolved']['environment'])
        restored['Config']['Labels']['com.docker.compose.config-hash'] = 'after'
        if post_drift == 'image': restored['Image'] = 'sha256:' + 'a' * 64
        elif post_drift == 'hash': restored['Config']['Labels']['com.docker.compose.config-hash'] = 'wrong'
        elif post_drift == 'secret':
            envset(restored, dict(base.environment(restored), TOKEN='changed'))
        elif post_drift == 'selection':
            envset(restored, dict(base.environment(restored), VLLM_EXTRA_ARGS=module.transform_args(ARGS, max_num_seqs=1)))
        if runtime['fail'] or post_drift == 'exited': restored['State'] = dict(Running=False, ExitCode=1)
        runtime['current'] = restored
        if runtime['fail']:
            runtime['fail'] = False
            raise SystemExit('synthetic Compose startup failure')
        return ''

    monkeypatch.setattr(module.base, 'run', run)
    return module, directory, original_path, runtime


def test_prepare_cli_private_recovery_never_rewrites_original_or_recaptures(baseline, tmp_path, monkeypatch, capsys):
    module, directory, original, runtime = cli_runtime(baseline, tmp_path, monkeypatch, prepare=True)
    before = original.read_bytes()
    module.main(['prepare', '--state-dir', str(directory)])
    state_path = directory / 'throughput.json'
    assert json.loads(state_path.read_text())['baseline_environment'] == {'VLLM_EXTRA_ARGS': ARGS}
    if os.name != 'nt': assert state_path.stat().st_mode & 0o777 == 0o600
    assert 'synthetic-secret' not in state_path.read_text()
    assert 'synthetic-secret' not in capsys.readouterr().out
    assert not any('up' in call for call in runtime['calls'])
    assert original.read_bytes() == before
    runtime['calls'].clear()
    with pytest.raises(SystemExit, match='recapture'): module.main(['prepare', '--state-dir', str(directory)])
    assert runtime['calls'] == []


@pytest.mark.parametrize('limit,profile', [(1, None), (2, None), (4, None), (1, 'decode'), (1, 'prefill')])
def test_cli_changes_only_vllm_and_restores_exact_original(baseline, tmp_path, monkeypatch, capsys, limit, profile):
    module, directory, original, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    before = original.read_bytes()
    args = ['switch', '--state-dir', str(directory), '--max-num-seqs', str(limit)]
    if profile: args += ['--profile', profile]
    module.main(args)
    actual = base.environment(runtime['current'])
    assert actual['TOKEN'] == 'synthetic-secret' and actual['VLLM_PLUGINS'] == 'sndr,existing-plugin'
    assert all(actual[name] == value for name, value in FLAGS.items())
    tokens = shlex.split(actual['VLLM_EXTRA_ARGS'])
    assert f'--max-num-seqs={limit}' in tokens
    assert ('--profiler-config' in tokens) is bool(profile)
    assert actual.get('VLLM_CUSTOM_SCOPES_FOR_PROFILING') == ('1' if profile else None)
    module.main(['rollback', '--state-dir', str(directory)])
    assert base.environment(runtime['current']) == base.environment(baseline[1])
    assert original.read_bytes() == before
    assert 'synthetic-secret' not in (directory / 'throughput-next.json').read_text()
    assert 'synthetic-secret' not in capsys.readouterr().out
    assert len([call for call in runtime['calls'] if 'up' in call]) == 2


@pytest.mark.parametrize('drift', ['resolved', 'image', 'hash', 'secret', 'selection', 'exited'])
def test_cli_preflight_or_post_inspection_detects_drift(baseline, tmp_path, monkeypatch, drift):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch, post_drift=drift)
    with pytest.raises(SystemExit):
        module.main(['switch', '--state-dir', str(directory), '--max-num-seqs', '4'])
    replacements = [call for call in runtime['calls'] if 'up' in call]
    assert len(replacements) == (0 if drift == 'resolved' else 1)


def test_failed_compose_start_is_inspected_and_can_explicitly_rollback(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch, up_failure=True)
    with pytest.raises(SystemExit, match='startup failed'):
        module.main(['switch', '--state-dir', str(directory), '--max-num-seqs', '4'])
    assert runtime['calls'][-1] == ['docker', 'inspect', 'compose-vllm-1']
    assert runtime['current']['State']['Running'] is False
    module.main(['rollback', '--state-dir', str(directory)])
    assert base.environment(runtime['current']) == base.environment(baseline[1])


def test_rollback_dollar_text_survives_compose_interpolation(baseline, tmp_path, monkeypatch):
    original, current, service, _ = baseline
    values = base.environment(current)
    text = ARGS + " --served-model-name 'literal$HOME${TOKEN}$$suffix'"
    values['VLLM_EXTRA_ARGS'] = text
    envset(current, values)
    service['environment'] = values.copy()
    original['environment_digests'] = base.protected_environment(values)
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    module.main(['switch', '--state-dir', str(directory), '--max-num-seqs', '2'])
    module.main(['rollback', '--state-dir', str(directory)])
    assert base.environment(runtime['current'])['VLLM_EXTRA_ARGS'] == text


def test_recovery_creation_requests_private_permissions_before_writing(tmp_path, monkeypatch):
    module = helper()
    original_open, original_chmod = module.os.open, module.os.chmod
    operations = []
    def private_open(path, flags, mode):
        operations.append(('open', mode, bool(flags & os.O_EXCL)))
        return original_open(path, flags, mode)
    def private_chmod(path, mode):
        operations.append(('chmod', mode))
        return original_chmod(path, mode)
    monkeypatch.setattr(module.os, 'open', private_open)
    monkeypatch.setattr(module.os, 'chmod', private_chmod)
    path = tmp_path / 'recovery.json'
    module.write_private(path, {'original': 'public text'}, exclusive=True)
    assert operations == [('open', 0o600, True), ('chmod', 0o600)]
    assert json.loads(path.read_text()) == {'original': 'public text'}
    with pytest.raises(FileExistsError): module.write_private(path, {}, exclusive=True)
    assert json.loads(path.read_text()) == {'original': 'public text'}


def test_valid_original_but_missing_recovery_still_blocks_before_docker(baseline, tmp_path, monkeypatch):
    module = helper()
    original = tmp_path / 'authoritative.json'
    original.write_text(json.dumps(baseline[0]))
    monkeypatch.setattr(module, 'ORIGINAL_STATE', original)
    monkeypatch.setattr(module.socket, 'gethostname', lambda: 'aquillm-dev2')
    monkeypatch.setattr(module.base, 'run', lambda *a, **k: pytest.fail('missing recovery touched Docker'))
    with pytest.raises(SystemExit): module.main(['rollback', '--state-dir', str(tmp_path)])


def test_unknown_h100_control_is_preserved_and_drift_is_protected(baseline):
    original, current, service, _ = baseline
    values = dict(base.environment(current), AQUILLM_H100_EXTRA_CONTROL='unchanged')
    envset(current, values)
    service['environment'] = values.copy()
    original['environment_digests'] = base.protected_environment(values)
    original['baseline_flags']['AQUILLM_H100_EXTRA_CONTROL'] = 'unchanged'
    state = prepared(baseline)
    override, process = helper().make_override(state, original, current, max_num_seqs=2, process_env={})
    assert override['services']['vllm']['environment']['AQUILLM_H100_EXTRA_CONTROL'] == '${AQUILLM_H100_CAPTURED_ENV_AQUILLM_H100_EXTRA_CONTROL}'
    assert process['AQUILLM_H100_CAPTURED_ENV_AQUILLM_H100_EXTRA_CONTROL'] == 'unchanged'
    values['AQUILLM_H100_EXTRA_CONTROL'] = 'changed'
    envset(current, values)
    with pytest.raises(SystemExit): helper().make_override(state, original, current, rollback=True, process_env={})


CANDIDATE = 'sha256:' + 'b' * 64


def candidate_image(baseline):
    image = copy.deepcopy(baseline[3])
    image['Id'] = CANDIDATE
    return image


def registered(baseline):
    return helper().authorize_image(prepared(baseline), baseline[0], baseline[1],
                                    baseline[3], candidate_image(baseline))


def test_registration_retains_immutable_recovery_capture(baseline):
    before = prepared(baseline)
    state = helper().authorize_image(before, baseline[0], baseline[1], baseline[3], candidate_image(baseline))
    assert state['approved_images'] == {IMAGE: base.digest(baseline[3]['Config']),
                                        CANDIDATE: base.digest(candidate_image(baseline)['Config'])}
    assert {key: value for key, value in state.items() if key != 'approved_images'} == {
        key: value for key, value in before.items() if key != 'approved_images'}
    assert before['approved_images'] == {IMAGE: base.digest(baseline[3]['Config'])}


@pytest.mark.parametrize('image', ['tag:latest', 'sha256:' + 'b' * 63, 'sha256:' + 'B' * 64, IMAGE])
def test_registration_requires_new_full_immutable_image(baseline, image):
    candidate = candidate_image(baseline)
    candidate['Id'] = image
    with pytest.raises(SystemExit):
        helper().authorize_image(prepared(baseline), baseline[0], baseline[1], baseline[3], candidate)


@pytest.mark.parametrize('field', ['Cmd', 'Entrypoint', 'Env', 'User', 'WorkingDir', 'Healthcheck',
                                  'ExposedPorts', 'Volumes', 'StopSignal', 'Shell'])
def test_registration_rejects_image_runtime_default_drift(baseline, field):
    candidate = candidate_image(baseline)
    candidate['Config'][field] = 'changed'
    with pytest.raises(SystemExit):
        helper().authorize_image(prepared(baseline), baseline[0], baseline[1], baseline[3], candidate)


@pytest.mark.parametrize('condition', ['exited', 'unhealthy', 'batch', 'profile', 'candidate', 'baseline-config'])
def test_registration_requires_exact_healthy_unmodified_original_baseline(baseline, condition):
    state = prepared(baseline)
    original, current, _, image = copy.deepcopy(baseline)
    if condition == 'exited': current['State']['Running'] = False
    elif condition == 'unhealthy': current['State']['Health'] = {'Status': 'unhealthy'}
    elif condition == 'candidate': current['Image'] = CANDIDATE
    elif condition == 'baseline-config': image['Config']['Cmd'] = ['changed']
    else:
        values = base.environment(current)
        values.update(helper().selected_environment(state, 2 if condition == 'batch' else 1,
                                                    'decode' if condition == 'profile' else None))
        envset(current, values)
    with pytest.raises(SystemExit): helper().authorize_image(state, original, current, image, candidate_image(baseline))


def test_registered_candidate_seq4_failed_container_restores_exact_base_and_environment(baseline):
    state = registered(baseline)
    original, current, _, _ = copy.deepcopy(baseline)
    override, _ = helper().make_override(state, original, current, max_num_seqs=4, image=CANDIDATE, process_env={})
    assert override['services']['vllm']['image'] == CANDIDATE
    current['Image'] = CANDIDATE
    values = base.environment(current)
    values.update(helper().selected_environment(state, 4))
    envset(current, values)
    current['State'] = {'Running': False}
    rollback, _ = helper().make_override(state, original, current, rollback=True, process_env={})
    assert rollback['services']['vllm']['image'] == IMAGE
    assert rollback['services']['vllm']['environment']['VLLM_EXTRA_ARGS'] == ARGS
    assert rollback['services']['vllm']['environment']['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] is None


def test_unregistered_candidate_and_candidate_profile_are_rejected(baseline):
    state = prepared(baseline)
    with pytest.raises(SystemExit):
        helper().make_override(state, baseline[0], baseline[1], image=CANDIDATE, process_env={})
    state = registered(baseline)
    with pytest.raises(SystemExit):
        helper().make_override(state, baseline[0], baseline[1], image=CANDIDATE, profile='decode', process_env={})


def test_legacy_state_migration_only_initializes_exact_baseline(baseline):
    state = prepared(baseline)
    state.pop('approved_images', None)
    state['schema_version'] = 1
    before = copy.deepcopy(state)
    migrated = helper().migrate_state(state, baseline[3])
    assert migrated['approved_images'] == {IMAGE: state['image_config_digest']}
    assert state == before
    with pytest.raises(SystemExit):
        helper().make_override(migrated, baseline[0], baseline[1], image=CANDIDATE, process_env={})
    altered = copy.deepcopy(baseline[3])
    altered['Config']['Env'] = ['changed']
    with pytest.raises(SystemExit): helper().migrate_state(state, altered)


def test_cli_registration_candidate_failure_and_exact_rollback(baseline, tmp_path, monkeypatch, capsys):
    module, directory, original, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    before = original.read_bytes()
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    assert not any('up' in call for call in runtime['calls'])
    state = json.loads((directory / 'throughput.json').read_text())
    assert CANDIDATE in state['approved_images']
    runtime['fail'] = True
    with pytest.raises(SystemExit, match='startup failed'):
        module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE, '--max-num-seqs', '4'])
    assert runtime['current']['Image'] == CANDIDATE
    module.main(['rollback', '--state-dir', str(directory)])
    assert runtime['current']['Image'] == IMAGE
    assert base.environment(runtime['current']) == base.environment(baseline[1])
    assert original.read_bytes() == before
    assert 'synthetic-secret' not in capsys.readouterr().out


def test_cli_candidate_config_change_blocks_before_compose(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    runtime['images'][CANDIDATE]['Config']['Cmd'] = ['changed']
    runtime['calls'].clear()
    with pytest.raises(SystemExit):
        module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE, '--max-num-seqs', '2'])
    assert not any('config' in call or 'up' in call for call in runtime['calls'])


def test_cli_unregistered_image_blocks_before_docker(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match='not been explicitly authorized'):
        module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE, '--max-num-seqs', '2'])
    assert runtime['calls'] == []


def test_registration_state_replace_failure_preserves_exact_prior_file(tmp_path, monkeypatch):
    module = helper()
    path = tmp_path / 'throughput.json'
    module.write_private(path, {'baseline': 'immutable'}, exclusive=True)
    prior = path.read_bytes()
    def refuse(*args): raise OSError('synthetic failed atomic replacement')
    monkeypatch.setattr(module.os, 'replace', refuse)
    with pytest.raises(OSError): module.write_private_atomic(path, {'candidate': 'registered'})
    assert path.read_bytes() == prior
    assert list(tmp_path.iterdir()) == [path]


def test_registration_protects_exact_json_types_in_healthcheck(baseline):
    baseline[3]['Config']['Healthcheck'] = {'Test': ['CMD', 'true'], 'Interval': 1}
    candidate = candidate_image(baseline)
    candidate['Config']['Healthcheck']['Interval'] = True
    with pytest.raises(SystemExit):
        helper().authorize_image(prepared(baseline), baseline[0], baseline[1], baseline[3], candidate)


def test_candidate_metadata_labels_bound_after_authorization(baseline):
    candidate = candidate_image(baseline)
    candidate['Config']['Labels'] = {'aquillm.candidate': 'mixed-gate-fix'}
    state = helper().authorize_image(prepared(baseline), baseline[0], baseline[1], baseline[3], candidate)
    helper().validate_image(state, candidate)
    candidate['Config']['Labels']['aquillm.candidate'] = 'other'
    with pytest.raises(SystemExit): helper().validate_image(state, candidate)


@pytest.mark.parametrize('drift', ['running-batch', 'running-profile', 'resolved', 'hash'])
def test_cli_failed_authorization_never_updates_recovery_or_starts_container(baseline, tmp_path, monkeypatch, drift):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch,
                                                post_drift='resolved' if drift == 'resolved' else None)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    path = directory / 'throughput.json'
    before = path.read_bytes()
    if drift.startswith('running-'):
        values = base.environment(runtime['current'])
        values.update(module.selected_environment(json.loads(before), 2 if drift == 'running-batch' else 1,
                                                  'decode' if drift == 'running-profile' else None))
        envset(runtime['current'], values)
    if drift == 'hash':
        current_run = module.base.run
        def wrong_hash(command, **kwargs):
            if command[-3:] == ['config', '--hash', 'vllm']: return 'vllm wrong\n'
            return current_run(command, **kwargs)
        monkeypatch.setattr(module.base, 'run', wrong_hash)
    with pytest.raises(SystemExit): module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    assert path.read_bytes() == before
    assert not any('up' in call for call in runtime['calls'])


def test_legacy_cli_baseline_switch_and_registration_preserve_capture(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    path = directory / 'throughput.json'
    old = json.loads(path.read_text())
    old.pop('approved_images')
    module.write_private(path, old)
    module.main(['switch', '--state-dir', str(directory), '--max-num-seqs', '1', '--profile', 'decode'])
    module.main(['rollback', '--state-dir', str(directory)])
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    registered_state = json.loads(path.read_text())
    assert {key: value for key, value in registered_state.items() if key != 'approved_images'} == old
    assert CANDIDATE in registered_state['approved_images']


def test_candidate_runtime_config_drift_blocks_rollback_before_compose(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    module.main(['switch', '--state-dir', str(directory), '--max-num-seqs', '2', '--image', CANDIDATE])
    runtime['images'][CANDIDATE]['Config']['Env'] = ['changed']
    runtime['calls'].clear()
    with pytest.raises(SystemExit): module.main(['rollback', '--state-dir', str(directory)])
    assert not any('config' in call or 'up' in call for call in runtime['calls'])


@pytest.mark.parametrize('image', ['', 0, False, 'candidate:latest'])
def test_override_rejects_invalid_explicit_image_instead_of_defaulting(baseline, image):
    with pytest.raises(SystemExit):
        helper().make_override(prepared(baseline), baseline[0], baseline[1], image=image, process_env={})


@pytest.mark.parametrize('limit,sizes', [(2, [5, 10, 20]), (4, [5, 10, 15, 20, 40])])
def test_mtp_graph_policy_adds_only_exact_capture_sizes(limit, sizes):
    module = helper()
    default = module.transform_args(ARGS, max_num_seqs=limit)
    tokens = shlex.split(module.transform_args(ARGS, max_num_seqs=limit, graph_policy='mtp'))
    assert tokens[:-2] == shlex.split(default)
    assert tokens[-2] == '--compilation-config'
    assert json.loads(tokens[-1]) == {'cudagraph_capture_sizes': sizes}
    assert module.transform_args(ARGS, max_num_seqs=limit, graph_policy='baseline') == default


@pytest.mark.parametrize('limit,profile,policy', [(1, None, 'mtp'), (2, 'decode', 'mtp'),
                                                (4, 'prefill', 'mtp'), (2, None, 'other')])
def test_mtp_graph_policy_rejects_sequence_profile_or_policy_mismatch(limit, profile, policy):
    with pytest.raises(SystemExit):
        helper().transform_args(ARGS, max_num_seqs=limit, profile=profile, graph_policy=policy)


@pytest.mark.parametrize('control', [
    '--compilation-config {}', '--compilation-config={}', '-cc {}', '-cc={}',
    '-c {}', '-c={}', '-c.cudagraph_capture_sizes [5,10,20]',
    '--config model.yaml', '--config=model.yaml',
    '--compilation_config {}', '--compilation-config.cudagraph_capture_sizes [5,10,20]',
    '--compilation {}', '--cudagraph-capture-sizes 5 10 20', '--cudagraph_capture_sizes 5 10 20',
    '--max-cudagraph-capture-size 20', '--max-cudagraph 20', '--cuda-graph-sizes 5 10 20',
    '--cudagraph-capture-sizes 5 --cudagraph-capture-sizes 10',
    '--compilation-config {broken',
])
def test_mtp_graph_policy_refuses_existing_or_ambiguous_graph_controls(control):
    with pytest.raises(SystemExit):
        helper().transform_args(ARGS + ' ' + control, max_num_seqs=2, graph_policy='mtp')


def test_default_policy_preserves_existing_compilation_settings():
    text = ARGS + " --compilation-config '{\"mode\":3}'"
    expected = shlex.split(text)
    expected[expected.index('--max-num-seqs=1')] = '--max-num-seqs=2'
    assert shlex.split(helper().transform_args(text, max_num_seqs=2)) == expected


@pytest.mark.parametrize('limit', [2, 4])
def test_mtp_candidate_failed_start_and_default_transition_restore_exact_original(baseline, tmp_path, monkeypatch, limit):
    module, directory, original_path, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    captured = (directory / 'throughput.json').read_bytes()
    authoritative = original_path.read_bytes()
    runtime['fail'] = True
    with pytest.raises(SystemExit, match='startup failed'):
        module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE,
                     '--max-num-seqs', str(limit), '--graph-policy', 'mtp'])
    assert runtime['current']['Image'] == CANDIDATE
    actual = base.environment(runtime['current'])
    assert actual['VLLM_EXTRA_ARGS'] == module.transform_args(ARGS, max_num_seqs=limit, graph_policy='mtp')
    assert actual['VLLM_PLUGINS'] == 'sndr,existing-plugin'
    module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE, '--max-num-seqs', '2'])
    assert '--compilation-config' not in shlex.split(base.environment(runtime['current'])['VLLM_EXTRA_ARGS'])
    module.main(['rollback', '--state-dir', str(directory)])
    assert runtime['current']['Image'] == IMAGE
    assert base.environment(runtime['current']) == base.environment(baseline[1])
    assert (directory / 'throughput.json').read_bytes() == captured
    assert original_path.read_bytes() == authoritative


@pytest.mark.parametrize('selection', [[], ['--image', IMAGE], ['--image', CANDIDATE, '--max-num-seqs', '1'],
                                      ['--image', CANDIDATE, '--profile', 'decode']])
def test_cli_mtp_bad_target_or_profile_precedes_docker(baseline, tmp_path, monkeypatch, selection):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    runtime['images'][CANDIDATE] = candidate_image(baseline)
    module.main(['authorize-image', '--state-dir', str(directory), '--image', CANDIDATE])
    runtime['calls'].clear()
    args = ['switch', '--state-dir', str(directory), '--graph-policy', 'mtp']
    if '--max-num-seqs' not in selection: args += ['--max-num-seqs', '2']
    with pytest.raises(SystemExit): module.main(args + selection)
    assert runtime['calls'] == []


@pytest.mark.parametrize('drift', ['baseline-image', 'wrong-seq', 'wrong-array', 'extra-field', 'duplicate-key', 'float-size', 'unknown-flag'])
def test_current_graph_policy_is_allowed_only_for_exact_candidate_array_and_limit(baseline, drift):
    state = registered(baseline)
    original, current, _, _ = copy.deepcopy(baseline)
    current['Image'] = CANDIDATE
    text = helper().transform_args(ARGS, max_num_seqs=4, graph_policy='mtp')
    if drift == 'baseline-image': current['Image'] = IMAGE
    elif drift == 'wrong-seq': text = text.replace('--max-num-seqs=4', '--max-num-seqs=2')
    elif drift == 'wrong-array': text = text.replace('[5,10,15,20,40]', '[5,10,20,40]')
    elif drift == 'extra-field': text = text.replace('{"cudagraph_capture_sizes":', '{"mode":3,"cudagraph_capture_sizes":')
    elif drift == 'duplicate-key': text = text.replace('{"cudagraph_capture_sizes":', '{"cudagraph_capture_sizes":[],"cudagraph_capture_sizes":')
    elif drift == 'float-size': text = text.replace('[5,10,15,20,40]', '[5.0,10,15,20,40]')
    elif drift == 'unknown-flag': text += ' --enforce-eager'
    envset(current, dict(base.environment(current), VLLM_EXTRA_ARGS=text))
    for rollback in (False, True):
        with pytest.raises(SystemExit): helper().make_override(state, original, current, rollback=rollback, process_env={})


def test_legacy_state_can_restore_registered_mtp_candidate_without_recapture(baseline):
    state = prepared(baseline)
    state.pop('approved_images')
    state = helper().authorize_image(state, baseline[0], baseline[1], baseline[3], candidate_image(baseline))
    current = copy.deepcopy(baseline[1])
    current['Image'] = CANDIDATE
    envset(current, dict(base.environment(current), VLLM_EXTRA_ARGS=helper().transform_args(ARGS, max_num_seqs=4, graph_policy='mtp')))
    override, _ = helper().make_override(state, baseline[0], current, rollback=True, process_env={})
    assert override['services']['vllm']['image'] == IMAGE
    assert override['services']['vllm']['environment']['VLLM_EXTRA_ARGS'] == ARGS


def test_original_compilation_config_remains_protected_and_blocks_mtp_only(baseline):
    original, current, service, _ = baseline
    text = ARGS + " --compilation-config '{\"mode\":3,\"custom_ops\":[\"+rms_norm\"]}'"
    values = dict(base.environment(current), VLLM_EXTRA_ARGS=text)
    envset(current, values)
    service['environment'] = values.copy()
    original['environment_digests'] = base.protected_environment(values)
    state = registered(baseline)
    before = copy.deepcopy(state)
    with pytest.raises(SystemExit, match='existing compilation'):
        helper().make_override(state, original, current, max_num_seqs=4, image=CANDIDATE,
                               graph_policy='mtp', process_env={})
    assert state == before
    override, _ = helper().make_override(state, original, current, max_num_seqs=4, image=CANDIDATE, process_env={})
    current['Image'] = CANDIDATE
    envset(current, dict(values, VLLM_EXTRA_ARGS=override['services']['vllm']['environment']['VLLM_EXTRA_ARGS']))
    rollback, _ = helper().make_override(state, original, current, rollback=True, process_env={})
    assert rollback['services']['vllm']['environment']['VLLM_EXTRA_ARGS'] == text


@pytest.mark.parametrize('scopes', [None, '', '0', '1'])
def test_mtp_rollback_restores_absent_extra_args_and_exact_scopes(baseline, scopes):
    original, current, service, _ = baseline
    values = base.environment(current)
    values.pop('VLLM_EXTRA_ARGS')
    if scopes is not None: values['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] = scopes
    envset(current, values)
    service['environment'] = values.copy()
    original['environment_digests'] = base.protected_environment(values)
    state = registered(baseline)
    current['Image'] = CANDIDATE
    envset(current, dict(values, **helper().selected_environment(state, 4, graph_policy='mtp')))
    current['State']['Running'] = False
    override, _ = helper().make_override(state, original, current, rollback=True, process_env={})
    selected = override['services']['vllm']['environment']
    assert selected['VLLM_EXTRA_ARGS'] is None
    assert selected['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] == scopes


def test_mtp_unregistered_candidate_rejected_before_docker(baseline, tmp_path, monkeypatch):
    module, directory, _, runtime = cli_runtime(baseline, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match='not been explicitly authorized'):
        module.main(['switch', '--state-dir', str(directory), '--image', CANDIDATE,
                     '--max-num-seqs', '4', '--graph-policy', 'mtp'])
    assert runtime['calls'] == []


def test_mtp_refuses_original_profiler_instead_of_silently_removing_it():
    config = json.dumps(helper().profiler_config('decode'))
    text = ARGS + ' --profiler-config ' + shlex.quote(config)
    with pytest.raises(SystemExit): helper().transform_args(text, max_num_seqs=2, graph_policy='mtp')
