import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dev_switch
import quality_bench
import serve_bench


def sse(*events):
    return io.BytesIO(b''.join(b'data: ' + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b'\n\n' for event in events))


class StreamTests(unittest.TestCase):
    def probe(self, *events):
        with patch.object(serve_bench, 'request', return_value=sse(*events)):
            return serve_bench.stream_request('http://unused', dict(max_tokens=4, ignore_eos=True))

    def test_truncation_is_failed_evidence(self):
        result = self.probe({'choices': [{'text': 'partial'}]})
        self.assertFalse(result['complete'])
        self.assertIsNotNone(result['error'])
        self.assertIsNone(result['aggregate_decode_seconds_per_token'])

    def test_server_error_is_failed_evidence(self):
        result = self.probe({'error': {'message': 'engine failure'}}, '[DONE]')
        self.assertFalse(result['complete'])
        self.assertIn('server_error', result['error'])

    def test_named_sse_error_event_is_failed_evidence(self):
        data = b'event: error\ndata: {"message":"engine failure"}\n\ndata: [DONE]\n'
        with patch.object(serve_bench, 'request', return_value=io.BytesIO(data)):
            result = serve_bench.stream_request('http://unused', dict(max_tokens=4, ignore_eos=True))
        self.assertFalse(result['complete'])
        self.assertEqual(result['error'], 'server_error')

    def test_role_chunk_does_not_set_ttft_and_burst_is_not_tokens(self):
        result = self.probe({'choices': [{'delta': {'role': 'assistant'}}]},
                            {'choices': [{'text': 'four tokens at once', 'finish_reason': 'length'}]},
                            {'usage': {'completion_tokens': 4}}, '[DONE]')
        self.assertTrue(result['complete'])
        self.assertEqual(result['nonempty_chunks'], 1)
        self.assertEqual(result['output_tokens'], 4)
        self.assertEqual(result['output_text'], 'four tokens at once')
        self.assertEqual(result['output_sha256'], hashlib.sha256(result['output_text'].encode()).hexdigest())
        self.assertIsNotNone(result['ttft_seconds'])
        empty = self.probe({'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': 'length'}]},
                           {'usage': {'completion_tokens': 4}}, '[DONE]')
        self.assertIsNone(empty['ttft_seconds'])

    def test_missing_or_wrong_usage_and_finish_fail(self):
        for usage in (None, {'completion_tokens': 3}, {'completion_tokens': True}):
            events = [{'choices': [{'text': 'x', 'finish_reason': 'length'}]}]
            if usage is not None:
                events.append({'usage': usage})
            self.assertFalse(self.probe(*events, '[DONE]')['complete'])
        self.assertFalse(self.probe({'choices': [{'text': 'x', 'finish_reason': 'stop'}]},
                                    {'usage': {'completion_tokens': 4}}, '[DONE]')['complete'])

    def test_transport_failure_returns_error_row(self):
        with patch.object(serve_bench, 'request', side_effect=OSError('transport broke')):
            result = serve_bench.stream_request('http://unused', dict(max_tokens=4, ignore_eos=True))
        self.assertFalse(result['complete'])
        self.assertIn('transport_error', result['error'])

    def test_transport_failure_after_partial_text_retains_failed_row(self):
        class BrokenStream(io.BytesIO):
            def __iter__(self):
                yield b'data: {"choices":[{"text":"partial"}]}\n'
                raise OSError('lost connection')
        with patch.object(serve_bench, 'request', return_value=BrokenStream()):
            result = serve_bench.stream_request('http://unused', dict(max_tokens=4, ignore_eos=True))
        self.assertFalse(result['complete'])
        self.assertEqual(result['nonempty_chunks'], 1)
        self.assertIsNone(result['aggregate_decode_seconds_per_token'])

    def test_done_without_terminal_choice_or_invalid_json_fails(self):
        for events in (({'choices': [{'text': 'x'}]}, {'usage': {'completion_tokens': 4}}, '[DONE]'),
                       ('{invalid', '[DONE]')):
            self.assertFalse(self.probe(*events)['complete'])

    def test_main_records_shape_metrics_outside_unchanged_timed_requests(self):
        events, payloads = [], []
        counter = 0
        def metrics(base):
            nonlocal counter
            events.append(('metrics', counter))
            snapshot = ['vllm:spec_decode_num_draft_tokens ' + str(counter)]
            counter += 1
            return snapshot
        def generate(base, payload):
            events.append(('request', len(payload['prompt'])))
            payloads.append(payload)
            return {'complete': True, 'error': None, 'output_tokens': 4, 'output_text': 'ok'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'serving.jsonl'
            argv = ['serve_bench.py', '--label', 'candidate-block2', '--output', str(output),
                    '--model', 'frozen-model', '--prompt-tokens', '512,32768', '--output-tokens', '4',
                    '--warmup', '1', '--repeats', '2']
            with patch.object(sys, 'argv', argv), patch.object(serve_bench, 'request', return_value=io.BytesIO(b'{"tokens":[1,2,3]}')), patch.object(serve_bench, 'speculation_metrics', side_effect=metrics), patch.object(serve_bench, 'stream_request', side_effect=generate), patch('builtins.print'):
                serve_bench.main()
            capture = json.loads(output.with_name('serving-candidate-block2-metrics.json').read_text())
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(events, [('metrics', 0), ('metrics', 1)] + [('request', 512)] * 3
                         + [('metrics', 2), ('metrics', 3)] + [('request', 32768)] * 3
                         + [('metrics', 4), ('metrics', 5)])
        self.assertEqual(capture['label'], 'candidate-block2')
        self.assertEqual(capture['before'], ['vllm:spec_decode_num_draft_tokens 0'])
        self.assertEqual(capture['after'], ['vllm:spec_decode_num_draft_tokens 5'])
        self.assertEqual([shape['prompt_tokens'] for shape in capture['shapes']], [512, 32768])
        for index, shape in enumerate(capture['shapes']):
            self.assertEqual(shape['before'], ['vllm:spec_decode_num_draft_tokens ' + str(1 + 2 * index)])
            self.assertEqual(shape['after'], ['vllm:spec_decode_num_draft_tokens ' + str(2 + 2 * index)])
            self.assertEqual(shape['requested_output_tokens'], 4)
            self.assertEqual(shape['input_sha256'], rows[index * 3]['input_sha256'])
            self.assertTrue(shape['interval_includes_warmups'])
        self.assertEqual([row['repeat'] for row in rows], [-1, 0, 1] * 2)
        self.assertEqual([row['warmup'] for row in rows], [True, False, False] * 2)
        for payload in payloads:
            self.assertEqual({key: value for key, value in payload.items() if key != 'prompt'},
                dict(model='frozen-model', max_tokens=4, temperature=0, seed=17, ignore_eos=True,
                     stream=True, stream_options={'include_usage': True}))
            expected = ([1, 2, 3] * (len(payload['prompt']) // 3 + 1))[:len(payload['prompt'])]
            self.assertEqual(payload['prompt'], expected)


class QualityTests(unittest.TestCase):
    def check(self, case, content, finish='stop'):
        return quality_bench.evaluate(case, {'choices': [{'message': {'content': content}, 'finish_reason': finish}]})

    def test_exact_answers_reject_sign_contradiction_and_extra_tokens(self):
        case = quality_bench.cases()[0]
        self.assertTrue(self.check(case, ' 730041\n')['passed'])
        for text in ('-730041', 'The answer is not 730041; it is 42.', '730041 42', '730041.'):
            self.assertFalse(self.check(case, text)['passed'])
        self.assertFalse(self.check(case, '730041', 'length')['passed'])
        self.assertFalse(self.check(case, '730041', None)['complete'])

    def test_word_case_normalizes_but_remembered_values_do_not(self):
        cases = quality_bench.cases()
        self.assertTrue(self.check(cases[-1], 'Yes')['passed'])
        self.assertFalse(self.check(cases[16], 'VIOLET-827')['passed'])

    def test_tools_require_exact_types_name_arguments_and_completion(self):
        case = quality_bench.cases()[8]
        call = dict(type='function', function=dict(name='record_measurement', arguments=json.dumps(case['expected'])))
        def evaluate(call, finish='tool_calls'):
            return quality_bench.evaluate(case, {'choices': [{'message': {'tool_calls': [call]}, 'finish_reason': finish}]})
        self.assertTrue(evaluate(call)['passed'])
        for bad in (dict(call, type='other'), dict(type='function', function=dict(name='record_measurement', arguments='{"label":"sample-0","value":17.0}')), {}):
            self.assertFalse(evaluate(bad)['passed'])
        self.assertFalse(evaluate(call, 'length')['passed'])
        self.assertFalse(quality_bench.evaluate(case, {'error': {'message': 'failed'}})['complete'])


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.current = {'Image': 'sha256:original', 'Config': {'Env': ['MODEL=x', 'TOKEN=secret'], 'Cmd': ['serve'], 'Labels': {'com.docker.compose.config-hash': 'verified'}}, 'HostConfig': {'Binds': ['/old:/cache']}, 'Mounts': [{'Source': '/old', 'Destination': '/cache'}]}
        self.service = {'image': 'original', 'environment': {'MODEL': 'x', 'TOKEN': 'secret'}, 'command': ['serve'], 'volumes': [{'source': '/old', 'target': '/cache'}]}
        self.legacy = {'image': 'sha256:original', 'project': 'compose', 'files': ['compose.yml'], 'working_dir': '/repo'}
        self.current['Config']['Labels'].update({'com.docker.compose.project': 'compose', 'com.docker.compose.project.working_dir': '/repo', 'com.docker.compose.project.config_files': 'compose.yml'})

    def prepare(self):
        return dev_switch.prepare_state(self.legacy, self.current, self.service, 'verified', verified=True)

    def test_new_env_and_unknown_aquillm_flag_rejected(self):
        state = self.prepare()
        for name in ('VLLM_REVISION', 'AQUILLM_H100_SECRET_NEW_FLAG'):
            service = dict(self.service, environment=dict(self.service['environment'], **{name: 'changed'}))
            with self.assertRaises(SystemExit):
                dev_switch.validate_configuration(state, self.current, service)

    def test_command_mount_and_runtime_drift_rejected(self):
        state = self.prepare()
        for service in (dict(self.service, command=['different']), dict(self.service, volumes=[{'source': '/new', 'target': '/cache'}])):
            with self.assertRaises(SystemExit):
                dev_switch.validate_configuration(state, self.current, service)
        current = dict(self.current, HostConfig={'Binds': ['/new:/cache']})
        with self.assertRaises(SystemExit):
            dev_switch.validate_configuration(state, current, self.service)

    def test_legacy_prepare_requires_original_image_and_verified_config(self):
        for current, digest, verified in ((dict(self.current, Image='candidate'), 'verified', True), (self.current, 'wrong', True), (self.current, 'verified', False)):
            with self.assertRaises(SystemExit):
                dev_switch.prepare_state(self.legacy, current, self.service, digest, verified=verified)
        state = self.prepare()
        self.assertNotIn('secret', json.dumps(state))
        self.assertEqual(state['schema_version'], 2)

    def test_prepared_baseline_cannot_be_recaptured_with_drift(self):
        state = self.prepare()
        with self.assertRaises(SystemExit):
            dev_switch.prepare_state(state, self.current, dict(self.service, command=['different']), 'verified', verified=True)
        current = json.loads(json.dumps(self.current))
        current['Config']['Env'].append('AQUILLM_H100_MTP_KERNEL=secret')
        with self.assertRaises(SystemExit):
            dev_switch.prepare_state(self.legacy, current, self.service, 'verified', verified=True)

    def test_rollback_uses_baseline_flags_and_rejects_candidate_env_drift(self):
        state = self.prepare()
        current = json.loads(json.dumps(self.current))
        current['Config']['Env'].append('AQUILLM_H100_MTP_KERNEL=fused')
        service = dict(self.service, environment=dict(self.service['environment'], AQUILLM_H100_MTP_KERNEL='fused'))
        dev_switch.validate_configuration(state, current, service)
        override, process = dev_switch.make_override(state, current, state['image'], rollback=True)
        self.assertIsNone(override['services']['vllm']['environment']['AQUILLM_H100_MTP_KERNEL'])
        self.assertNotIn('secret', json.dumps(override))
        current['Config']['Env'].append('VLLM_REVISION=changed')
        with self.assertRaises(SystemExit):
            dev_switch.make_override(state, current, state['image'], rollback=True)

    def main_probe(self, state, current, resolved, action='rollback', candidate_env=None, baseline_env=None):
        calls = []
        restored = json.loads(json.dumps(self.current))
        def docker(args, env=None, input=None):
            calls.append(args)
            if args[:2] == ['docker', 'inspect']:
                return json.dumps([restored if any('up' in c for c in calls) else current])
            if args[:3] == ['docker', 'image', 'inspect']:
                values = candidate_env if args[3] == 'candidate' else baseline_env
                image_config = {'Env': values} if values is not None else {}
                return json.dumps([{'Id': 'sha256:original', 'Config': image_config}])
            if 'config' in args:
                if '--hash' in args:
                    return 'vllm verified\n'
                return json.dumps({'services': {'vllm': resolved}})
            return ''
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / '.config/aquillm/h100-performance'
            config.mkdir(parents=True)
            (config / 'baseline.json').write_text(json.dumps(state))
            argv = ['dev_switch.py', action] + (['--image', 'candidate'] if action == 'switch' else [])
            with patch.object(dev_switch.Path, 'home', return_value=home), patch.object(dev_switch.socket, 'gethostname', return_value='aquillm-dev2'), patch.object(dev_switch, 'run', side_effect=docker), patch.object(sys, 'argv', argv), patch('builtins.print'):
                try:
                    dev_switch.main()
                except SystemExit as exc:
                    return calls, str(exc), None
                override = json.loads((config / 'next.json').read_text()) if (config / 'next.json').exists() else None
                return calls, None, override

    def test_main_rejects_new_environment_and_mount_before_compose_up(self):
        state = self.prepare()
        for resolved in (dict(self.service, environment=dict(self.service['environment'], VLLM_REVISION='new')), dict(self.service, command=['different']), dict(self.service, volumes=[{'source': '/new', 'target': '/cache'}])):
            calls, error, _ = self.main_probe(state, self.current, resolved)
            self.assertIsNotNone(error)
            self.assertFalse(any('up' in call for call in calls))

    def test_main_clean_rollback_restores_baseline_flags(self):
        current = json.loads(json.dumps(self.current))
        current['Image'] = 'sha256:candidate'
        current['Config']['Env'].append('AQUILLM_H100_MTP_KERNEL=fused')
        calls, error, override = self.main_probe(self.prepare(), current, self.service)
        self.assertIsNone(error)
        self.assertTrue(any('up' in call for call in calls))
        self.assertEqual(override['services']['vllm']['image'], 'sha256:original')
        self.assertIsNone(override['services']['vllm']['environment']['AQUILLM_H100_MTP_KERNEL'])

    def test_main_legacy_rollback_rejected_before_up(self):
        calls, error, _ = self.main_probe(self.legacy, self.current, self.service)
        self.assertIn('migrated', error)
        self.assertFalse(any('up' in call for call in calls))

    def test_candidate_image_new_protected_env_rejected_before_up(self):
        for env in (['VLLM_REVISION=new-revision'], ['AQUILLM_H100_UNKNOWN_FLAG=new']):
            calls, error, _ = self.main_probe(self.prepare(), self.current, self.service,
                                             action='switch', candidate_env=env)
            self.assertIsNotNone(error)
            self.assertFalse(any('up' in call for call in calls))

    def test_candidate_image_changed_protected_env_value_rejected_before_up(self):
        calls, error, _ = self.main_probe(self.prepare(), self.current, self.service,
            action='switch', candidate_env=['VLLM_REVISION=changed'], baseline_env=['VLLM_REVISION=original'])
        self.assertIn('Image protected environment', error)
        self.assertFalse(any('up' in call for call in calls))

    def test_canonical_hash_resolves_env_file_without_writing_credentials(self):
        raw = json.dumps({'services': {'vllm': dict(self.service, environment={'TOKEN': 'secret$$value'})}})
        with patch.object(dev_switch, 'run', return_value='vllm verified\n') as command:
            value = dev_switch.canonical_hash(self.legacy, raw, {'TOKEN': 'secret$value'})
        self.assertEqual(value, 'verified')
        args, kwargs = command.call_args
        self.assertIn('-', args[0])
        self.assertNotIn('compose.yml', args[0])
        self.assertEqual(kwargs['input'], raw)
        self.assertEqual(dev_switch.parse_config(raw)['services']['vllm']['environment']['TOKEN'], 'secret$value')

    def test_prepare_main_accepts_verified_resolved_hash_but_rejects_drift(self):
        raw = json.dumps({'services': {'vllm': self.service}})
        for canonical in ('verified', 'changed'):
            calls = []
            def docker(args, env=None, input=None):
                calls.append((args, input))
                if args[:2] == ['docker', 'inspect']:
                    return json.dumps([self.current])
                if '--hash' in args:
                    # Old raw config hash differs; only resolved stdin matches.
                    return 'vllm ' + (canonical if input == raw else 'unresolved-env-file-hash') + '\n'
                return raw
            with tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                config = home / '.config/aquillm/h100-performance'
                config.mkdir(parents=True)
                state_path = config / 'baseline.json'
                state_path.write_text(json.dumps(self.legacy))
                with patch.object(dev_switch.Path, 'home', return_value=home), patch.object(dev_switch.socket, 'gethostname', return_value='aquillm-dev2'), patch.object(dev_switch, 'run', side_effect=docker), patch.object(sys, 'argv', ['dev_switch.py', 'prepare', '--verify-current-baseline']), patch('builtins.print') as output:
                    if canonical == 'verified':
                        dev_switch.main()
                        self.assertEqual(json.loads(state_path.read_text())['schema_version'], 2)
                    else:
                        with self.assertRaises(SystemExit):
                            dev_switch.main()
                        self.assertEqual(json.loads(state_path.read_text()), self.legacy)
                    self.assertNotIn('secret', str(output.call_args_list))
                self.assertNotIn('secret', state_path.read_text())
                self.assertFalse(any('up' in args for args, _ in calls))

    def test_prefill_switch_independent_of_mtp_and_rollback_restores_original(self):
        current = json.loads(json.dumps(self.current))
        current['Config']['Env'].append('AQUILLM_H100_PREFILL=0')
        state = dev_switch.prepare_state(self.legacy, current, self.service, 'verified', verified=True)
        override, _ = dev_switch.make_override(state, current, 'candidate', prefill='1')
        flags = override['services']['vllm']['environment']
        self.assertEqual(flags['AQUILLM_H100_MTP_KERNEL'], 'baseline')
        self.assertEqual(flags['AQUILLM_H100_PREFILL'], '1')
        current['Config']['Env'][-1] = 'AQUILLM_H100_PREFILL=1'
        rollback, _ = dev_switch.make_override(state, current, state['image'], rollback=True, prefill='1')
        self.assertEqual(rollback['services']['vllm']['environment']['AQUILLM_H100_PREFILL'], '0')
        with self.assertRaises(SystemExit):
            dev_switch.make_override(state, current, 'candidate', prefill='arbitrary')

    def test_candidate_runtime_and_gdn_switch_preserve_prefill_and_rollback(self):
        state = self.prepare()
        override, _ = dev_switch.make_override(state, self.current, 'candidate',
            prefill='1', runtime_profile='flashinfer-0.6.18', gdn='flashinfer')
        values = override['services']['vllm']['environment']
        self.assertEqual(values['AQUILLM_H100_PREFILL'], '1')
        self.assertEqual(values['AQUILLM_H100_RUNTIME_PROFILE'], 'flashinfer-0.6.18')
        self.assertEqual(values['AQUILLM_H100_GDN'], 'flashinfer')
        current = json.loads(json.dumps(self.current))
        current['Config']['Env'].extend([
            'AQUILLM_H100_RUNTIME_PROFILE=flashinfer-0.6.18', 'AQUILLM_H100_GDN=flashinfer'])
        rollback, _ = dev_switch.make_override(state, current, state['image'], rollback=True)
        self.assertIsNone(rollback['services']['vllm']['environment']['AQUILLM_H100_RUNTIME_PROFILE'])
        self.assertIsNone(rollback['services']['vllm']['environment']['AQUILLM_H100_GDN'])
        for kwargs in ({'runtime_profile': 'anything'}, {'gdn': 'anything'}):
            with self.assertRaises(SystemExit):
                dev_switch.make_override(state, self.current, 'candidate', **kwargs)

    def mount_fixture(self):
        current = json.loads(json.dumps(self.current))
        current['Config']['Cmd'] = ['serve', '--model', 'model']
        current['Mounts'].extend([{'Type': 'volume', 'Name': 'compile-cache', 'Source': '/volumes/compile-cache', 'Destination': '/compile', 'RW': True},
                                 {'Type': 'bind', 'Source': '/script.py', 'Destination': '/script.py', 'RW': False}])
        current['HostConfig']['Binds'].append('/script.py:/script.py:ro')
        current['HostConfig']['Mounts'] = [{'Type': 'volume', 'Source': 'cache', 'Target': '/cache'},
                                          {'Type': 'volume', 'Source': 'compile-cache', 'Target': '/compile'}]
        return current

    def test_native_gdn_baseline_switch_requires_pair_and_preserves_rollback(self):
        state = self.prepare()
        override, _ = dev_switch.make_override(state,self.current,'native-image',
            prefill='1',runtime_profile='native-gdn-baseline',gdn='native-fp16')
        values = override['services']['vllm']['environment']
        self.assertEqual(values['AQUILLM_H100_PREFILL'],'1')
        self.assertEqual(values['AQUILLM_H100_RUNTIME_PROFILE'],'native-gdn-baseline')
        self.assertEqual(values['AQUILLM_H100_GDN'],'native-fp16')
        current = json.loads(json.dumps(self.current))
        current['Config']['Env'].extend(['AQUILLM_H100_RUNTIME_PROFILE=native-gdn-baseline',
                                       'AQUILLM_H100_GDN=native-fp16'])
        rollback, _ = dev_switch.make_override(state,current,state['image'],rollback=True)
        self.assertIsNone(rollback['services']['vllm']['environment']['AQUILLM_H100_GDN'])
        self.assertIsNone(rollback['services']['vllm']['environment']['AQUILLM_H100_RUNTIME_PROFILE'])
        for kwargs in ({'gdn':'native-fp16'},{'runtime_profile':'native-gdn-baseline'},
            {'gdn':'native-fp16','runtime_profile':'flashinfer-0.6.18'}):
            with self.assertRaises(SystemExit):
                dev_switch.make_override(state,self.current,'native-image',**kwargs)

    def test_runtime_digest_ignores_only_mount_collection_order(self):
        current = self.mount_fixture()
        reordered = json.loads(json.dumps(current))
        reordered['Mounts'].reverse()
        reordered['HostConfig']['Binds'].reverse()
        reordered['HostConfig']['Mounts'].reverse()
        self.assertEqual(dev_switch.runtime_digest(current), dev_switch.runtime_digest(reordered))
        for change in ('mount', 'volume_name', 'rw', 'command', 'log_config'):
            changed = json.loads(json.dumps(reordered))
            if change == 'mount':
                changed['Mounts'][0]['Source'] = '/changed'
            elif change == 'volume_name':
                changed['Mounts'][1]['Name'] = 'different-data'
            elif change == 'rw':
                changed['Mounts'][0]['RW'] = True
            elif change == 'command':
                changed['Config']['Cmd'].reverse()
            else:
                changed['HostConfig']['LogConfig'] = {'Type': 'none'}
            self.assertNotEqual(dev_switch.runtime_digest(current), dev_switch.runtime_digest(changed))

    def test_legacy_digest_migration_requires_exact_mount_permutation_proof(self):
        current = self.mount_fixture()
        state = dev_switch.prepare_state(self.legacy, current, self.service, 'verified', verified=True)
        state.pop('runtime_digest_format', None)
        state['runtime_digest'] = dev_switch.runtime_digest(current, canonical=False)
        reordered = json.loads(json.dumps(current))
        reordered['Mounts'].reverse()
        reordered['HostConfig']['Binds'].reverse()
        reordered['HostConfig']['Mounts'].reverse()
        dev_switch.validate_configuration(state, reordered, self.service)
        migrated = dev_switch.prepare_state(state, reordered, self.service, 'verified', verified=True)
        self.assertEqual(migrated['runtime_digest_format'], 'mount-order-v1')
        self.assertEqual(migrated['runtime_digest'], dev_switch.runtime_digest(current))
        for change in ('mount', 'command'):
            drifted = json.loads(json.dumps(reordered))
            if change == 'mount':
                drifted['Mounts'][0]['Source'] = '/changed'
            else:
                drifted['Config']['Cmd'].reverse()
            with self.assertRaises(SystemExit):
                dev_switch.prepare_state(state, drifted, self.service, 'verified', verified=True)

    def test_legacy_digest_permutation_search_is_bounded_and_unknown_format_fails(self):
        current = self.mount_fixture()
        state = dev_switch.prepare_state(self.legacy, current, self.service, 'verified', verified=True)
        state.pop('runtime_digest_format', None)
        state['runtime_digest'] = 'f' * 64
        current['Mounts'] = [{'Source': str(index), 'Destination': '/' + str(index)} for index in range(9)]
        with self.assertRaises(SystemExit):
            dev_switch.validate_configuration(state, current, self.service)
        current = self.mount_fixture()
        state = dev_switch.prepare_state(self.legacy, current, self.service, 'verified', verified=True)
        state['runtime_digest_format'] = 'future-unknown-format'
        with self.assertRaises(SystemExit):
            dev_switch.validate_configuration(state, current, self.service)


if __name__ == '__main__':
    unittest.main()
