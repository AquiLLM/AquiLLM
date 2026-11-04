"""Bounded development batching on explicitly registered immutable images.

Only EXTRA_ARGS and custom-scope presence/values vary in the environment. Private
recovery retains their original text; all other runtime records stay protected.
Profiling and rollback always use the exact restored baseline image.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import socket
import uuid

import dev_switch as base

PREFILL_IMAGE = 'sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7'
ORIGINAL_STATE = Path('/home/exouser/.config/aquillm/flashinfer-upgrade/baseline.json')
AFFECTED = frozenset(('VLLM_EXTRA_ARGS', 'VLLM_CUSTOM_SCOPES_FOR_PROFILING'))
PREFILL_FLAGS = dict(AQUILLM_H100_PREFILL='1', AQUILLM_H100_MTP_KERNEL='baseline',
                     AQUILLM_H100_SPLIT_POLICY='baseline', AQUILLM_H100_GDN='baseline')
IMAGE_DEFAULTS = ('Cmd', 'Entrypoint', 'Env', 'User', 'WorkingDir', 'Healthcheck',
                  'ExposedPorts', 'Volumes', 'StopSignal', 'Shell')
MTP_GRAPH_SIZES = {2: (5, 10, 20), 4: (5, 10, 15, 20, 40)}


def immutable_image(image):
    if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise SystemExit('An explicit full immutable sha256 image ID is required')
    return image


def approved_images(state):
    # Legacy recovery can select only its already captured baseline, never a
    # candidate. Migration adds the mapping after inspecting that exact image.
    images = state.get('approved_images')
    if images is None and state.get('schema_version') == 1:
        images = {PREFILL_IMAGE: state.get('image_config_digest')}
    if (not isinstance(images, dict) or images.get(PREFILL_IMAGE) != state.get('image_config_digest')
            or any(not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                   for digest in images.values())):
        raise SystemExit('Approved immutable image recovery is malformed')
    for image in images:
        immutable_image(image)
    return images


def validate_image(state, image):
    identifier = immutable_image(image.get('Id'))
    if approved_images(state).get(identifier) != base.digest(image.get('Config')):
        raise SystemExit('Unregistered image or approved image configuration drift')


def migrate_state(state, baseline_image):
    if baseline_image.get('Id') != PREFILL_IMAGE:
        raise SystemExit('Migration requires the exact baseline image')
    validate_image(state, baseline_image)
    migrated = copy.deepcopy(state)
    migrated['approved_images'] = dict(approved_images(state))
    return migrated


def authorize_image(state, original, current, baseline_image, candidate_image):
    validate_configuration(state, original, current)
    validate_image(state, baseline_image)
    if (current.get('Image') != PREFILL_IMAGE or baseline_image.get('Id') != PREFILL_IMAGE
            or not current.get('State', {}).get('Running')
            or current.get('State', {}).get('Health', {}).get('Status', 'healthy') != 'healthy'
            or affected_environment(base.environment(current)) != state['baseline_environment']):
        raise SystemExit('Image authorization requires the exact running healthy original baseline')
    identifier = immutable_image(candidate_image.get('Id'))
    if identifier == PREFILL_IMAGE:
        raise SystemExit('Candidate authorization requires a distinct immutable image')
    config, baseline_config = candidate_image.get('Config'), baseline_image.get('Config')
    if not isinstance(config, dict) or not isinstance(baseline_config, dict):
        raise SystemExit('Image runtime defaults are unavailable')
    if base.digest({field: config[field] for field in IMAGE_DEFAULTS if field in config}) != base.digest(
            {field: baseline_config[field] for field in IMAGE_DEFAULTS if field in baseline_config}):
        raise SystemExit('Candidate image protected runtime defaults differ from baseline')
    digest = base.digest(config)
    registered = migrate_state(state, baseline_image)
    previous = registered['approved_images'].get(identifier)
    if previous is not None and previous != digest:
        raise SystemExit('Previously authorized candidate image configuration drift')
    # Labels may describe candidate provenance. Every Config field, including
    # labels, is bound by this digest; authorization does not qualify its code.
    registered['approved_images'][identifier] = digest
    return registered


def profiler_config(profile):
    if profile not in ('decode', 'prefill'):
        raise SystemExit('Only bounded decode/prefill profiler modes are supported')
    wait, warmup, active = (2, 2, 32) if profile == 'decode' else (0, 0, 1)
    return dict(profiler='torch', torch_profiler_dir='/tmp/aquillm-profile/' + profile,
        ignore_frontend=True, torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        torch_profiler_dump_cuda_time_total=False, wait_iterations=wait,
        warmup_iterations=warmup, active_iterations=active, max_iterations=active)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _tokens(text):
    if not isinstance(text, str):
        raise SystemExit('EXTRA_ARGS must be text')
    try:
        tokens = shlex.split(text)
    except ValueError:
        raise SystemExit('Malformed EXTRA_ARGS quoting; values suppressed') from None
    seen, selected = set(), {}
    for index, token in enumerate(tokens):
        if not token.startswith('--'):
            continue
        option, separator, value = token.partition('=')
        if not re.fullmatch(r'--[a-zA-Z][a-zA-Z0-9_-]*', option):
            raise SystemExit('Malformed argument flag; values suppressed')
        canonical = option.replace('_', '-')
        if canonical in seen:
            raise SystemExit('Duplicate argument flag; values suppressed')
        seen.add(canonical)
        if canonical not in ('--max-num-seqs', '--profiler-config'):
            # argparse abbreviation must not alias either controlled option.
            if any(target.startswith(canonical) for target in ('--max-num-seqs', '--profiler-config')):
                raise SystemExit('Abbreviated controlled flags are forbidden')
            continue
        if canonical != option:
            raise SystemExit('Controlled flags require their canonical spelling')
        if not separator:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith('-'):
                raise SystemExit('Controlled flag value is missing')
            value = tokens[index + 1]
        if canonical == '--max-num-seqs':
            if value not in ('1', '2', '4'):
                raise SystemExit('Sequence limit must be exactly 1, 2 or 4')
        else:
            try:
                config = json.loads(value, object_pairs_hook=_unique_object)
            except (ValueError, TypeError):
                raise SystemExit('Malformed profiler JSON; values suppressed') from None
            approved = False
            if isinstance(config, dict):
                for mode in ('decode', 'prefill'):
                    expected = profiler_config(mode)
                    approved |= config == expected and all(type(config[key]) is type(value) for key, value in expected.items())
            if not approved:
                raise SystemExit('Profiler paths/options must match a bounded approved mode')
        selected[canonical] = (index, bool(separator), value)
    return tokens, selected


def _has_graph_control(tokens):
    # Pinned EngineArgs accepts -cc, JSON dotted keys, and top-level sizing
    # overrides. A YAML --config can also carry compilation options. Refuse
    # every spelling/prefix rather than merge unknown configuration.
    controls = ('--compilation-config', '--cudagraph-capture-sizes',
                '--max-cudagraph-capture-size', '--cuda-graph-sizes', '--config')
    for token in tokens:
        option = token.split('=', 1)[0].replace('_', '-')
        if option in ('-c', '-cc') or option.startswith(('-c.', '-cc.')):
            return True
        if option.startswith('--') and any(
                target.startswith(option) or option.startswith(target + '.') for target in controls):
            return True
    return False


def transform_args(original, *, max_num_seqs, profile=None, graph_policy='baseline'):
    if type(max_num_seqs) is not int or max_num_seqs not in (1, 2, 4):
        raise SystemExit('Sequence limit must be exactly 1, 2 or 4')
    if graph_policy not in ('baseline', 'mtp'):
        raise SystemExit('Only baseline or bounded MTP graph policies are supported')
    if graph_policy == 'mtp' and (max_num_seqs not in MTP_GRAPH_SIZES or profile is not None):
        raise SystemExit('MTP graph policy requires seq2/4 without profiling')
    if profile is not None and (profile not in ('decode', 'prefill') or max_num_seqs != 1):
        raise SystemExit('Bounded profiler runs require max-num-seqs=1')
    tokens, selected = _tokens(original)
    if graph_policy == 'mtp' and '--profiler-config' in selected:
        raise SystemExit('MTP graph policy refuses existing profiler configuration')
    if graph_policy == 'mtp' and _has_graph_control(tokens):
        raise SystemExit('MTP graph policy refuses existing compilation/graph controls')
    result, skip = [], set()
    for name, (index, joined, _) in selected.items():
        skip.add(index)
        if not joined:
            skip.add(index + 1)
    for index, token in enumerate(tokens):
        if index in skip:
            if '--max-num-seqs' in selected and index == selected['--max-num-seqs'][0]:
                joined = selected['--max-num-seqs'][1]
                result += [f'--max-num-seqs={max_num_seqs}'] if joined else ['--max-num-seqs', str(max_num_seqs)]
            continue
        result.append(token)
    if '--max-num-seqs' not in selected:
        result += ['--max-num-seqs', str(max_num_seqs)]
    if profile is not None:
        result += ['--profiler-config', json.dumps(profiler_config(profile), separators=(',', ':'))]
    if graph_policy == 'mtp':
        # EngineArgs 2dfaae752 --compilation-config accepts this dataclass JSON.
        # Only capture sizes change; maximum remains 20/40, and C1 retains 5.
        result += ['--compilation-config', json.dumps(
            {'cudagraph_capture_sizes': MTP_GRAPH_SIZES[max_num_seqs]}, separators=(',', ':'))]
    return shlex.join(result)


def affected_environment(values):
    return {name: str(values[name]) for name in AFFECTED if name in values and values[name] is not None}


def protected_environment(values):
    # Unlike dev_switch's image experiments, every H100/plugin flag is protected.
    return {name: base.digest(str(value)) for name, value in values.items()
            if name not in AFFECTED and value is not None}


def validate_identity(original, current):
    labels = current['Config'].get('Labels', {})
    if (current.get('Name') != '/compose-vllm-1' or labels.get('com.docker.compose.service') != 'vllm'
            or labels.get('com.docker.compose.project') != original.get('project')
            or labels.get('com.docker.compose.project.working_dir') != original.get('working_dir')):
        raise SystemExit('Development model/Compose identity drift')


def validate_original(original):
    if (original.get('schema_version') != 2 or original.get('image') != PREFILL_IMAGE
            or not isinstance(original.get('baseline_flags'), dict)
            or any(original['baseline_flags'].get(name) != value for name, value in PREFILL_FLAGS.items())):
        raise SystemExit('Verified authoritative prefill baseline state is required')


def prepare_state(original, current, resolved, compose_hash, image):
    validate_original(original)
    validate_identity(original, current)
    if current.get('Image') != PREFILL_IMAGE or image.get('Id') != PREFILL_IMAGE:
        raise SystemExit('Preparation requires the exact baseline image')
    if not current.get('State', {}).get('Running') or current.get('State', {}).get('Health', {}).get('Status', 'healthy') != 'healthy':
        raise SystemExit('Preparation requires a running healthy baseline')
    if compose_hash != current['Config']['Labels'].get('com.docker.compose.config-hash'):
        raise SystemExit('Running resolved Compose configuration is unverified')
    base.validate_configuration(original, current, resolved)
    values = base.environment(current)
    if any(values.get(name) != value for name, value in PREFILL_FLAGS.items()):
        raise SystemExit('Preparation requires prefill and baseline H100 controls')
    if any(values.get(name) != value for name, value in original.get('baseline_flags', {}).items()):
        raise SystemExit('Authoritative baseline flag values/presence differ')
    _, selected = _tokens(values.get('VLLM_EXTRA_ARGS', ''))
    if '--profiler-config' in selected or ('--max-num-seqs' in selected and selected['--max-num-seqs'][2] != '1'):
        raise SystemExit('Preparation requires the unprofiled single-sequence baseline')
    return dict(schema_version=1, image=PREFILL_IMAGE, project=original['project'],
        working_dir=original['working_dir'], files=list(original['files']),
        original_state_digest=base.digest(original), baseline_environment=affected_environment(values),
        environment_digests=protected_environment(values), baseline_flags=dict(original['baseline_flags']),
        service_digest=base.service_digest(resolved), runtime_digest_format='mount-order-v1',
        runtime_digest=base.runtime_digest(current), image_config_digest=base.digest(image['Config']),
        approved_images={PREFILL_IMAGE: base.digest(image['Config'])})


def selected_environment(state, max_num_seqs=1, profile=None, *, rollback=False, graph_policy='baseline'):
    original = state['baseline_environment']
    if rollback:
        return dict(original)
    values = dict(original)
    values['VLLM_EXTRA_ARGS'] = transform_args(original.get('VLLM_EXTRA_ARGS', ''), max_num_seqs=max_num_seqs,
                                              profile=profile, graph_policy=graph_policy)
    if profile is not None:
        values['VLLM_CUSTOM_SCOPES_FOR_PROFILING'] = '1'
    return values


def normalize_environment(state, values, *, image=PREFILL_IMAGE):
    if protected_environment(values) != state['environment_digests']:
        raise SystemExit('Protected environment/plugin/MTP flag drift; recovery refused')
    affected = affected_environment(values)
    allowed = [state['baseline_environment']]
    allowed += [selected_environment(state, limit) for limit in (1, 2, 4)]
    allowed += [selected_environment(state, 1, mode) for mode in ('decode', 'prefill')]
    # Existing original compilation settings remain protected under baseline
    # policy. They are never merged or replaced by the MTP experiment.
    baseline_tokens, baseline_selected = _tokens(state['baseline_environment'].get('VLLM_EXTRA_ARGS', ''))
    if (image != PREFILL_IMAGE and image in approved_images(state)
            and '--profiler-config' not in baseline_selected and not _has_graph_control(baseline_tokens)):
        allowed += [selected_environment(state, limit, graph_policy='mtp') for limit in MTP_GRAPH_SIZES]
    if affected not in allowed:
        raise SystemExit('Running/resolved sequence/profiler settings are outside the bounded experiment')
    normalized = {name: value for name, value in values.items() if name not in AFFECTED and value is not None}
    normalized.update(state['baseline_environment'])
    return normalized


def validate_configuration(state, original, current, resolved=None):
    validate_original(original)
    if (state.get('schema_version') != 1 or state.get('image') != PREFILL_IMAGE
            or state.get('original_state_digest') != base.digest(original)
            or not isinstance(state.get('baseline_environment'), dict)
            or set(state['baseline_environment']) - AFFECTED
            or state.get('baseline_flags') != original.get('baseline_flags')):
        raise SystemExit('Throughput recovery does not match the authoritative baseline')
    validate_identity(original, current)
    if current.get('Image') not in approved_images(state):
        raise SystemExit('Image drift; only baseline or explicitly registered images are supported')
    normalized = copy.deepcopy(current)
    values = normalize_environment(state, base.environment(current), image=current['Image'])
    if current['Image'] != PREFILL_IMAGE:
        _, selected = _tokens(base.environment(current).get('VLLM_EXTRA_ARGS', ''))
        if '--profiler-config' in selected:
            raise SystemExit('Profiling is restricted to the baseline image')
    normalized['Config']['Env'] = [f'{name}={value}' for name, value in values.items()]
    if any(values.get(name) != value for name, value in state['baseline_flags'].items()):
        raise SystemExit('Baseline flag drift')
    if not base.runtime_matches(state, normalized):
        raise SystemExit('Protected runtime/command/mount drift')
    service = None
    if resolved is not None:
        if resolved.get('image') not in approved_images(state) or base.service_digest(resolved) != state['service_digest']:
            raise SystemExit('Resolved image/service configuration drift')
        service = copy.deepcopy(resolved)
        service['environment'] = normalize_environment(state, resolved.get('environment', {}), image=resolved['image'])
        if resolved['image'] != PREFILL_IMAGE and '--profiler-config' in _tokens(resolved.get('environment', {}).get('VLLM_EXTRA_ARGS', ''))[1]:
            raise SystemExit('Profiling is restricted to the baseline image')
    if base.protected_environment(values) != original.get('environment_digests') or not base.runtime_matches(original, normalized):
        raise SystemExit('Authoritative original environment/runtime drift')
    if service is not None:
        base.validate_configuration(original, normalized, service)


def make_override(state, original, current, *, max_num_seqs=1, profile=None, rollback=False, image=None,
                  process_env=None, graph_policy='baseline'):
    validate_configuration(state, original, current)
    target = PREFILL_IMAGE if rollback or image is None else image
    if immutable_image(target) not in approved_images(state):
        raise SystemExit('Requested image has not been explicitly authorized')
    if profile is not None and target != PREFILL_IMAGE:
        raise SystemExit('Profiling is restricted to the baseline image')
    if graph_policy == 'mtp' and (rollback or target == PREFILL_IMAGE):
        raise SystemExit('MTP graph policy requires an explicitly registered candidate image')
    values = base.environment(current)
    inherited = {name: '${' + base.PREFIX + name + '}' for name in values if name not in AFFECTED}
    selected = selected_environment(state, max_num_seqs, profile, rollback=rollback, graph_policy=graph_policy)
    # Override literals must survive Compose interpolation; parse_config restores
    # config's reusable escaping before comparisons with the exact runtime text.
    inherited.update({name: selected[name].replace('$', '$$') if name in selected else None for name in AFFECTED})
    process = dict(os.environ if process_env is None else process_env)
    process.update({base.PREFIX + name: value for name, value in values.items()})
    return {'services': {'vllm': {'image': target, 'environment': inherited}}}, process


def write_private(path, value, *, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            fd = None
            output.write(json.dumps(value, indent=2) + '\n')
    finally:
        if fd is not None:
            os.close(fd)


def write_private_atomic(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        write_private(temporary, value, exclusive=True)
        with temporary.open('r+b') as contents:
            os.fsync(contents.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(path, *, private=False):
    if not path.is_file() or path.is_symlink():
        raise SystemExit('Required recovery state is missing or not a regular file')
    if private and os.name != 'nt' and path.stat().st_mode & 0o077:
        raise SystemExit('Recovery state must have private mode 0600')
    try:
        state = json.loads(path.read_text())
    except (ValueError, OSError):
        raise SystemExit('Recovery state is unreadable or malformed; values suppressed') from None
    if not isinstance(state, dict):
        raise SystemExit('Recovery state must be an object')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'authorize-image', 'switch', 'rollback'))
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--max-num-seqs', type=int, choices=(1, 2, 4))
    parser.add_argument('--profile', choices=('decode', 'prefill'))
    parser.add_argument('--image')
    parser.add_argument('--graph-policy', choices=('baseline', 'mtp'), default='baseline')
    args = parser.parse_args(argv)
    if socket.gethostname() != 'aquillm-dev2':
        raise SystemExit('This experiment is restricted to the authorized 254 development host')
    if args.action == 'switch' and args.max_num_seqs is None:
        parser.error('switch requires --max-num-seqs')
    if args.action != 'switch' and (args.max_num_seqs is not None or args.profile is not None):
        parser.error('sequence/profiler selections are switch-only')
    if args.profile is not None and args.max_num_seqs != 1:
        parser.error('profiler runs require --max-num-seqs 1')
    if args.graph_policy == 'mtp' and (args.action != 'switch' or args.max_num_seqs not in MTP_GRAPH_SIZES
            or args.profile is not None or args.image is None or args.image == PREFILL_IMAGE):
        parser.error('MTP graphs require switch seq2/4, explicit registered candidate image and no profiler')
    if args.action == 'authorize-image' and args.image is None:
        parser.error('authorize-image requires --image')
    if args.image is not None:
        immutable_image(args.image)
        if args.action not in ('authorize-image', 'switch'):
            parser.error('image selection is authorization/switch-only')
        if args.profile is not None and args.image != PREFILL_IMAGE:
            parser.error('profiler runs require the baseline image')
    if not args.state_dir.is_absolute():
        parser.error('--state-dir must be an absolute private recovery directory')
    state_path = args.state_dir / 'throughput.json'
    if state_path.resolve() == ORIGINAL_STATE.resolve():
        raise SystemExit('Authoritative baseline state may not be overwritten')
    original = read_state(ORIGINAL_STATE)
    validate_original(original)
    if args.action == 'prepare' and state_path.exists():
        raise SystemExit('Recovery already exists; candidate recapture is forbidden')
    state = None if args.action == 'prepare' else read_state(state_path, private=True)
    target = args.image or PREFILL_IMAGE
    if state is not None and args.action == 'switch' and target not in approved_images(state):
        raise SystemExit('Requested image has not been explicitly authorized')
    current = json.loads(base.run(['docker', 'inspect', 'compose-vllm-1']))[0]
    validate_identity(original, current)
    if state is not None:
        validate_configuration(state, original, current)
    image = json.loads(base.run(['docker', 'image', 'inspect', PREFILL_IMAGE]))[0]
    if image.get('Id') != PREFILL_IMAGE or (state is not None and base.digest(image['Config']) != state['image_config_digest']):
        raise SystemExit('Pinned baseline image configuration drift')
    if state is not None:
        state = migrate_state(state, image)
        # Prove all current/target image Config digests before Compose mutation.
        for identifier in {current['Image'], target if args.action == 'switch' else PREFILL_IMAGE} - {PREFILL_IMAGE}:
            inspected = json.loads(base.run(['docker', 'image', 'inspect', identifier]))[0]
            if inspected.get('Id') != identifier:
                raise SystemExit('Inspected immutable image identity differs')
            validate_image(state, inspected)
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == 'prepare':
        labels = current['Config']['Labels']
        command = base.compose_command(original, labels['com.docker.compose.project.config_files'].split(','))
        process = dict(os.environ, **{base.PREFIX + name: value for name, value in base.environment(current).items()})
        raw = base.run(command + ['config', '--format', 'json'], env=process)
        resolved = base.parse_config(raw)['services']['vllm']
        state = prepare_state(original, current, resolved, base.canonical_hash(original, raw, process), image)
        write_private(state_path, state, exclusive=True)
    elif args.action == 'authorize-image':
        candidate = json.loads(base.run(['docker', 'image', 'inspect', target]))[0]
        if candidate.get('Id') != target:
            raise SystemExit('Inspected candidate image identity differs')
        registered = authorize_image(state, original, current, image, candidate)
        labels = current['Config']['Labels']
        command = base.compose_command(original, labels['com.docker.compose.project.config_files'].split(','))
        process = dict(os.environ, **{base.PREFIX + name: value for name, value in base.environment(current).items()})
        raw = base.run(command + ['config', '--format', 'json'], env=process)
        resolved = base.parse_config(raw)['services']['vllm']
        validate_configuration(state, original, current, resolved)
        if (resolved.get('image') != PREFILL_IMAGE
                or affected_environment(resolved.get('environment', {})) != state['baseline_environment']
                or base.canonical_hash(original, raw, process) != labels.get('com.docker.compose.config-hash')):
            raise SystemExit('Image authorization requires the verified original Compose baseline')
        write_private_atomic(state_path, registered)
    else:
        rollback = args.action == 'rollback'
        override, process = make_override(state, original, current,
            max_num_seqs=args.max_num_seqs or 1, profile=args.profile, rollback=rollback, image=target,
            graph_policy=args.graph_policy)
        target = override['services']['vllm']['image']
        override_path = args.state_dir / 'throughput-next.json'
        write_private(override_path, override)
        command = base.compose_command(state) + ['-f', str(override_path)]
        raw = base.run(command + ['config', '--format', 'json'], env=process)
        resolved = base.parse_config(raw)['services']['vllm']
        validate_configuration(state, original, current, resolved)
        expected = selected_environment(state, args.max_num_seqs or 1, args.profile, rollback=rollback,
                                        graph_policy=args.graph_policy)
        if affected_environment(resolved.get('environment', {})) != expected:
            raise SystemExit('Resolved sequence/profiler selection differs from the requested operation')
        if resolved.get('image') != target:
            raise SystemExit('Resolved image differs from the requested operation')
        expected_hash = base.canonical_hash(state, raw, process)
        failure = None
        try:
            base.run(command + ['up', '-d', '--no-deps', '--no-build', 'vllm'], env=process)
        except SystemExit as error:
            failure = error
        # State health is deliberately not a rollback precondition: failed-start
        # containers still retain the inherited protected environment for recovery.
        restored = json.loads(base.run(['docker', 'inspect', 'compose-vllm-1']))[0]
        validate_configuration(state, original, restored, resolved)
        if (restored.get('Image') != target or affected_environment(base.environment(restored)) != expected
                or restored['Config']['Labels'].get('com.docker.compose.config-hash') != expected_hash):
            raise SystemExit('Post-operation exact configuration verification failed; invoke rollback')
        if failure is not None or not restored.get('State', {}).get('Running'):
            raise SystemExit('Model startup failed; retained recovery permits explicit rollback') from None
    print(json.dumps(dict(action=args.action, image=target, verified=True,
                         max_num_seqs=args.max_num_seqs, profile=args.profile, graph_policy=args.graph_policy)), flush=True)


if __name__ == '__main__':
    main()
