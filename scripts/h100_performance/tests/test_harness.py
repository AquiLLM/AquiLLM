import io
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

    def main_probe(self, state, current, resolved, action='rollback'):
        calls = []
        restored = json.loads(json.dumps(self.current))
        def docker(args, env=None):
            calls.append(args)
            if args[:2] == ['docker', 'inspect']:
                return json.dumps([restored if any('up' in c for c in calls) else current])
            if args[:3] == ['docker', 'image', 'inspect']:
                return json.dumps([{'Id': 'sha256:original', 'Config': {}}])
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
            with patch.object(dev_switch.Path, 'home', return_value=home), patch.object(dev_switch.socket, 'gethostname', return_value='aquillm-dev2'), patch.object(dev_switch, 'run', side_effect=docker), patch.object(sys, 'argv', ['dev_switch.py', action]), patch('builtins.print'):
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


if __name__ == '__main__':
    unittest.main()
