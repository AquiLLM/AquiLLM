import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exposition(running=0, waiting=0, drafts=8, accepted=6, preemptions=0):
    return '\n'.join([
        '# TYPE vllm:num_requests_running gauge',
        f'vllm:num_requests_running{{engine="0",model_name="synthetic"}} {running}',
        f'vllm:num_requests_waiting{{engine="0",model_name="synthetic"}} {waiting}',
        'vllm:kv_cache_usage_perc{engine="0",model_name="synthetic"} 0.5',
        f'vllm:num_preemptions_total{{engine="0",model_name="synthetic"}} {preemptions}',
        f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="synthetic"}} {drafts / 4}',
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="synthetic"}} {drafts}',
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="synthetic"}} {accepted}',
        'vllm:spec_decode_num_accepted_tokens_created{engine="0"} 1791663000',
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} {accepted}',
    ])


class ThroughputContractTests(unittest.TestCase):
    def benchmark(self):
        self.assertTrue((ROOT / 'throughput_bench.py').exists(), 'Concurrent benchmark is not implemented')
        return importlib.import_module('throughput_bench')

    def test_mixed_work_is_frozen_balanced_and_not_client_concurrency_dependent(self):
        b = self.benchmark()
        work = b.freeze_workload([11, 22, 33], 'synthetic', 'mixed', 8)
        self.assertEqual([r['prompt_tokens'] for r in work], [512, 8192, 32768, 36864] * 2)
        self.assertEqual([r['requested_output_tokens'] for r in work], [256] * 8)
        self.assertEqual(len({r['id'] for r in work}), 8)
        self.assertEqual(work[0]['payload']['prompt'][:7], [11, 22, 33, 11, 22, 33, 11])
        self.assertEqual(work[0]['input_sha256'], work[4]['input_sha256'])
        for row in work:
            self.assertEqual(len(row['payload']['prompt']), row['prompt_tokens'])
            self.assertTrue(row['payload']['ignore_eos'])
        # A new independently constructed client arm must send exactly the same bytes.
        self.assertEqual(json.dumps(work, sort_keys=True), json.dumps(
            b.freeze_workload([11, 22, 33], 'synthetic', 'mixed', 8), sort_keys=True))

    def test_fixed_short_and_long_token_budgets(self):
        b = self.benchmark()
        for name, prompt, output in [('short', 512, 1024), ('long', 8192, 512)]:
            work = b.freeze_workload([7], 'synthetic', name, 8)
            self.assertEqual([(r['prompt_tokens'], r['requested_output_tokens']) for r in work],
                             [(prompt, output)] * 8)

    def test_empty_or_noninteger_seed_and_unbalanced_mixed_are_rejected(self):
        b = self.benchmark()
        for seed in ([], [True], [-1], ['1']):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                b.freeze_workload(seed, 'synthetic', 'short', 8)
        with self.assertRaises(ValueError):
            b.freeze_workload([1], 'synthetic', 'mixed', 3)

    def test_prometheus_labels_do_not_double_count_created_or_position_counters(self):
        b = self.benchmark()
        m = b.parse_metrics(exposition(running=2, waiting=3))
        self.assertEqual(m['running'], 2)
        self.assertEqual(m['waiting'], 3)
        self.assertEqual(m['draft_tokens'], 8)
        self.assertEqual(m['accepted_tokens'], 6)
        self.assertEqual(m['cache_usage'], 0.5)
        self.assertEqual(m['preemptions'], 0)
        # Sum active workers; cache usage is the maximum worker ratio, not a sum.
        extra = exposition(running=1).replace('engine="0"', 'engine="1"')
        multi = b.parse_metrics(exposition(running=2) + '\n' + extra)
        self.assertEqual(multi['running'], 3)
        self.assertEqual(multi['cache_usage'], 0.5)

    def test_missing_nonfinite_negative_and_duplicate_metrics_are_rejected(self):
        b = self.benchmark()
        for text in (exposition().replace('vllm:num_requests_waiting', 'unrelated'),
                     exposition().replace(' 0.5', ' NaN'),
                     exposition(running=-1), exposition() + '\n' + exposition()):
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError):
                b.parse_metrics(text)

    def test_reordered_duplicate_labels_cannot_inflate_observed_running(self):
        b = self.benchmark()
        duplicate = 'vllm:num_requests_running{model_name="synthetic",engine="0"} 2'
        with self.assertRaises(ValueError):
            b.parse_metrics(exposition(running=2) + '\n' + duplicate)

    def fixture(self):
        b = self.benchmark()
        work = b.freeze_workload([1, 2], 'synthetic', 'short', 2)
        rows = []
        for index, item in enumerate(work):
            text = f'output-{index}'
            rows.append(dict(id=item['id'], input_sha256=item['input_sha256'], complete=True,
                             error=None, stream_done=True, finish_reason='length',
                             ttft_seconds=0.2 + index * 0.2, total_seconds=2.0,
                             output_tokens=1024, output_text=text,
                             output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             usage=dict(prompt_tokens=512, completion_tokens=1024)))
        before = b.parse_metrics(exposition())
        after = b.parse_metrics(exposition(drafts=16, accepted=12))
        samples = [dict(captured_at='2026-10-10T20:00:00+00:00',
                        values=b.parse_metrics(exposition(running=2, waiting=1)))]
        return b, work, rows, before, after, samples

    def test_summary_uses_wall_time_and_proves_sampled_running_overlap(self):
        b, work, rows, before, after, samples = self.fixture()
        report = b.summarize_repeat(work, rows, 3.0, before, after, samples, [])
        self.assertTrue(report['complete'])
        self.assertAlmostEqual(report['output_tokens_per_second'], 2048 / 3)
        self.assertAlmostEqual(report['median_ttft_seconds'], 0.3)
        self.assertEqual(report['p95_ttft_seconds'], 0.4)
        self.assertEqual(report['instrumentation']['max_running'], 2)
        self.assertTrue(report['instrumentation']['observed_running_gt_one'])
        self.assertEqual(report['speculation']['draft_tokens'], 8)
        self.assertEqual(report['speculation']['accepted_tokens'], 6)
        self.assertEqual(report['speculation']['acceptance_rate'], 0.75)
        self.assertEqual(report['results'], rows)

    def test_partial_wrong_token_usage_hash_and_invalid_times_publish_no_speed(self):
        b, work, rows, before, after, samples = self.fixture()
        variants = []
        for key, value in [('complete', False), ('error', 'server_error'),
                           ('stream_done', False), ('output_tokens', 1023),
                           ('total_seconds', float('nan')), ('ttft_seconds', 3),
                           ('output_sha256', 'wrong'), ('input_sha256', 'wrong')]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            variants.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]['usage']['prompt_tokens'] = 511
        variants.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]['usage'] = ['invalid']
        variants.append(changed)
        variants += [rows[:1], [rows[0], rows[0]]]
        for invalid in variants:
            with self.subTest(invalid=str(invalid)[:100]):
                report = b.summarize_repeat(work, invalid, 3, before, after, samples, [])
                self.assertFalse(report['complete'])
                self.assertIsNone(report['output_tokens_per_second'])
                self.assertTrue(report['validation_errors'])

    def test_instrumentation_failure_or_counter_reset_invalidates_performance(self):
        b, work, rows, before, after, samples = self.fixture()
        for changed_after, changed_samples, errors in (
                (after, samples, ['metrics_timeout']), (after, [], []),
                ({**after, 'draft_tokens': 4}, samples, []), (None, samples, [])):
            report = b.summarize_repeat(work, rows, 3, before, changed_after, changed_samples, errors)
            self.assertFalse(report['complete'])
            self.assertIsNone(report['output_tokens_per_second'])
            self.assertFalse(report['instrumentation']['valid'])
        for wall in (0, -1, math.inf):
            self.assertFalse(b.summarize_repeat(work, rows, wall, before, after, samples, [])['complete'])
        self.assertFalse(b.summarize_repeat(work, rows, 1, before, after, samples, [])['complete'])

    def test_quality_suite_keeps_exact_oracles_and_disjoint_long_expected_values(self):
        b = self.benchmark()
        work = b.quality_workload('synthetic')
        self.assertEqual(len(work), 38)
        self.assertEqual(len({r['case']['id'] for r in work}), 38)
        expected = [json.dumps(r['case']['expected'], sort_keys=True) for r in work]
        self.assertEqual(len(set(expected)), 38)
        for row in work:
            case = row['case']
            self.assertFalse(row['payload']['chat_template_kwargs']['enable_thinking'])
            if 'tools' in case:
                message = dict(content='', tool_calls=[dict(type='function', function=dict(
                    name='record_measurement', arguments=json.dumps(case['expected'])))])
                finish = 'tool_calls'
            else:
                message, finish = dict(content=case['expected']), 'stop'
            result = b.evaluate(case, dict(choices=[dict(message=message, finish_reason=finish)]))
            self.assertTrue(result['passed'])
        for row in work[32:]:
            self.assertTrue(row['long_context'])
            self.assertIn(str(row['case']['expected']) if isinstance(row['case']['expected'], str)
                          else row['case']['expected']['label'],
                          ' '.join(x['content'] for x in row['case']['messages']))

    def test_settled_snapshot_rejects_nonidle_or_changing_server_counters(self):
        b = self.benchmark()
        self.assertTrue(callable(getattr(b, 'settled_metrics', None)), 'Settled metrics snapshot is missing')
        idle = dict(values=b.parse_metrics(exposition()), captured_at='first')
        changing = dict(values=b.parse_metrics(exposition(drafts=12)), captured_at='second')
        running = dict(values=b.parse_metrics(exposition(running=1)), captured_at='second')
        with patch.object(b.time, 'sleep'):
            for other in (changing, running):
                with patch.object(b, 'fetch_metrics', side_effect=[idle, other]), self.assertRaises(ValueError):
                    b.settled_metrics('http://unused', 6)
            with patch.object(b, 'fetch_metrics', side_effect=[idle, idle]):
                stable = b.settled_metrics('http://unused', 6)
                self.assertEqual(stable['values']['draft_tokens'], 8)
                self.assertEqual(stable['settle_seconds'], 6)
                self.assertEqual(len(stable['idle_stability_samples']), 2)


class LocalServingTests(unittest.TestCase):
    """Real HTTP, threads, metrics parsing, SSE and JSONL; no model or GPU imports."""
    def setUp(self):
        import throughput_bench
        self.b = throughput_bench
        state = self.state = dict(running=0, drafts=8, accepted=6, bad_metrics=False,
                                 bad_usage=False, bad_quality=False, seen=[])
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, body, content_type='application/json'):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != '/metrics':
                    self.send_error(404)
                    return
                with lock:
                    text = exposition(running=state['running'], drafts=state['drafts'], accepted=state['accepted'])
                if state['bad_metrics']:
                    text = text.replace('vllm:num_requests_waiting', 'unrelated')
                self.reply(text.encode(), 'text/plain')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    self.reply(json.dumps(dict(tokens=[11, 22, 33])).encode())
                    return
                if self.path == '/v1/chat/completions':
                    text = ' '.join(m['content'] for m in payload['messages'])
                    if 'tools' in payload:
                        label = re.search(r'(?:sample-\d+|concurrent-long-\d+)', text)[0]
                        value = int(re.search(r'(?:integer value|integer) (\d+)', text)[1])
                        message = dict(role='assistant', content=None, tool_calls=[dict(type='function',
                            function=dict(name='record_measurement', arguments=json.dumps(dict(label=label, value=value))))])
                        finish = 'tool_calls'
                    else:
                        number = re.search(r'archive key for project Cedar-\d+ is (\d+)', text)
                        violet = re.search(r'(?:violet-\d+|concurrent-violet-9182)', text)
                        literal_answers = {
                            'Compute 17 + 26. Reply with only the number.': '43',
                            'Compute 12 times 8. Reply with only the number.': '96',
                            'What is the capital of France? Reply with one word.': 'Paris',
                            'Which is larger, 19 or 23? Reply with the larger number only.': '23',
                            'A box has 40 balls. Remove 13, then add 8. How many? Reply with only the number.': '35',
                            'Translate the English word cat into Spanish. Reply with one word.': 'gato',
                            'How many sides does a hexagon have? Reply with only the number.': '6',
                            'If all robins are birds and Pip is a robin, is Pip a bird? Reply yes or no.': 'yes',
                        }
                        answer = number[1] if number else violet[0] if violet else literal_answers[text]
                        if state['bad_quality'] and answer == '730041':
                            answer = '900001'  # A valid answer to a different concurrent request.
                        message, finish = dict(role='assistant', content=answer), 'stop'
                    self.reply(json.dumps(dict(choices=[dict(message=message, finish_reason=finish)],
                                               usage=dict(prompt_tokens=42000 if len(text) > 100000 else 1777,
                                                          completion_tokens=7))).encode())
                    return
                if self.path != '/v1/completions':
                    self.send_error(404)
                    return
                with lock:
                    state['running'] += 1
                    state['seen'].append(payload)
                try:
                    time.sleep(0.26)  # Allows the independent 0.2-second poll to observe overlap.
                    usage = dict(prompt_tokens=len(payload['prompt']), completion_tokens=payload['max_tokens'])
                    if state['bad_usage']:
                        usage['prompt_tokens'] -= 1
                    events = [dict(choices=[dict(text='synthetic output', finish_reason='length')]), dict(usage=usage)]
                    body = b''.join(b'data: ' + json.dumps(x).encode() + b'\n\n' for x in events)
                    body += b'data: [DONE]\n\n'
                    with lock:
                        state['drafts'] += 4
                        state['accepted'] += 3
                    self.reply(body, 'text/event-stream')
                finally:
                    with lock:
                        state['running'] -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def invoke(self, destination, concurrency, *extra):
        argv = ['throughput_bench.py', '--mode', 'performance', '--base-url', self.base,
                '--model', 'synthetic', '--concurrency', str(concurrency), '--label', 'local',
                '--output', str(destination), '--requests', '4', '--repeats', '1', '--warmup', '0', *extra]
        argv += ['--metrics-settle-seconds', '1']
        with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit) as result:
            self.b.main()
        return result.exception.code, [json.loads(x) for x in destination.read_text().splitlines()]

    def test_real_concurrent_streams_preserve_frozen_identity_and_prove_running_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            code1, rows1 = self.invoke(Path(directory) / 'c1.jsonl', 1)
            code2, rows2 = self.invoke(Path(directory) / 'c2.jsonl', 2)
        self.assertEqual((code1, code2), (0, 0))
        one, two = rows1[0], rows2[0]
        self.assertEqual(one['frozen_identity'], two['frozen_identity'])
        self.assertEqual(one['total_output_tokens'], 4096)
        self.assertEqual(one['instrumentation']['max_running'], 1)
        self.assertEqual(two['instrumentation']['max_running'], 2)
        self.assertTrue(two['instrumentation']['observed_running_gt_one'])
        self.assertEqual(two['speculation']['draft_tokens'], 16)
        self.assertEqual(two['speculation']['accepted_tokens'], 12)
        self.assertEqual(len(two['results']), 4)
        self.assertEqual(self.state['seen'][:4], self.state['seen'][4:])

    def test_bad_metrics_and_wrong_server_prompt_counts_fail_cli_without_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            for flag in ('bad_metrics', 'bad_usage'):
                self.state[flag] = True
                code, rows = self.invoke(Path(directory) / (flag + '.jsonl'), 2)
                self.assertEqual(code, 1)
                self.assertFalse(rows[0]['complete'])
                self.assertIsNone(rows[0]['output_tokens_per_second'])
                self.state[flag] = False

    def test_full_concurrent_warmup_is_saved_but_excluded_from_counter_and_wall_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            code, rows = self.invoke(Path(directory) / 'warm.jsonl', 4, '--warmup', '1', '--workload', 'mixed')
        self.assertEqual(code, 0)
        row = rows[0]
        self.assertEqual(len(row['warmup_results']), 4)
        self.assertEqual(len(self.state['seen']), 8)
        self.assertEqual(row['total_output_tokens'], 1024)
        self.assertEqual(row['speculation']['draft_tokens'], 16)
        self.assertFalse(row['speculation']['interval_includes_warmups'])
        self.assertEqual(row['metrics_settle_seconds'], 1)
        self.assertTrue(row['instrumentation']['observed_running_gt_one'])

    def test_quality_cli_validates_all_oracles_and_detects_cross_request_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            code, rows = self.invoke(Path(directory) / 'quality.jsonl', 4, '--mode', 'quality')
            self.assertEqual(code, 0)
            self.assertEqual(len(rows), 38)
            self.assertTrue(all(r['passed'] and r['complete'] for r in rows))
            self.assertTrue(all('output_tokens_per_second' not in r and 'seconds' not in r for r in rows))
            self.assertTrue(all(r['long_context_exercised'] for r in rows[32:]))
            self.assertEqual(rows[32]['message']['content'], '900001')
            self.assertEqual(rows[0]['frozen_identity'], rows[-1]['frozen_identity'])
            self.state['bad_quality'] = True
            code, rows = self.invoke(Path(directory) / 'bad-quality.jsonl', 4, '--mode', 'quality')
            self.assertEqual(code, 1)
            self.assertFalse(rows[0]['passed'])

    def test_cli_refuses_to_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.jsonl'
            path.write_text('original evidence')
            with self.assertRaises(FileExistsError):
                self.invoke(path, 1)
            self.assertEqual(path.read_text(), 'original evidence')


if __name__ == '__main__':
    unittest.main()
