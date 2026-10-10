import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("allocator_analysis", Path(__file__).resolve().parents[1] / "allocator_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class Tests(unittest.TestCase):
    def test_three_boot_units_give_coarse_assumption_based_sign_flip_sensitivity(self):
        result = analysis.uncertainty([1.1, 1.1, 1.1])
        self.assertEqual(result["paired_boots"], 3)
        self.assertEqual(result["paired_sign_flip_sensitivity_two_sided_p"], .25)
        self.assertAlmostEqual(result["improvement_percent"], 10)
        self.assertEqual(analysis.uncertainty([1.1] * 30)["status"], "missing")

    def test_variable_boot_effects_widen_primary_interval(self):
        result = analysis.uncertainty([1.2, 1., .9])
        self.assertLess(result["block_t_95_percent"][0], 0)
        self.assertGreater(result["block_t_95_percent"][1], 0)

    def test_absent_data_remains_pending(self):
        with tempfile.TemporaryDirectory() as name:
            result = analysis.build(Path(name), 2)
        self.assertEqual(len(result["serving_errors"]), 6)
        self.assertFalse(result["protocol_screen"]["gain_at_least_5_and_block_t_interval_excludes_zero"])
        self.assertEqual(result["mixed_boot_uncertainty"]["status"], "missing")

    def test_warmups_and_errors_do_not_enter_measured_latency(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for role in ("system", "mimalloc"):
                for block in range(1, 4):
                    rows = []
                    for repeat in (-1, 0, 1):
                        row = dict(label=f"{role}-block{block}", input_sha256="a" * 64, prompt_tokens=512,
                            requested_output_tokens=4, output_tokens=4, output_sha256="b" * 64,
                            usage={"completion_tokens":4}, repeat=repeat, warmup=repeat < 0,
                            complete=True, error=None, stream_done=True, finish_reason="length",
                            ttft_seconds=999 if repeat < 0 else 1., total_seconds=1000 if repeat < 0 else 2.,
                            aggregate_decode_seconds_per_token=1/3)
                        if role == "mimalloc" and block == 1 and repeat == 1:
                            row.update(complete=False, error="missing_done", stream_done=False, total_seconds=200.)
                        rows.append(row)
                    (directory / f"h100-allocator-{role}-block{block}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            result = analysis.build(directory, 2)
        candidate = result["shapes"][0]["blocks"]["mimalloc-block1"]
        self.assertEqual(candidate["requests"], 1)
        self.assertEqual(candidate["ttft_ms"]["median"], 1000)
        self.assertEqual(candidate["total_ms"]["median"], 2000)
        self.assertTrue(any(error["reason"] == "incomplete_or_invalid_requests" for error in result["serving_errors"]))

    def test_process_report_suppresses_environments_and_unrelated_text(self):
        result = analysis.process_memory(dict(processes=[dict(status="VmRSS:\t4096 kB\nVmHWM:\t8192 kB\n",
            maps="/opt/mimalloc/lib/libmimalloc.so.2", environment={"TOKEN":"never-copy-this"})]))
        self.assertEqual(result["memory_fields"]["vmrss_kb"]["sum"], 4096)
        self.assertEqual(result["processes_with_mapped_mimalloc"], 1)
        self.assertNotIn("never-copy-this", json.dumps(result))

    def test_actual_process_schema_checks_api_env_and_engine_library_mapping(self):
        records = [dict(pid=11, role="api", configured_allocator="mimalloc", observed_env_allocator="mimalloc",
            observed_env_pythonmalloc="default", process_environment_reliable=True,
            libraries=["/opt/mimalloc/lib/libmimalloc.so.2"], status={"VmHWM":"3908012 kB", "VmRSS":"2557800 kB", "Threads":"28"},
            memory={"Rss":"2557800 kB", "Pss":"2314906 kB", "Private_Dirty":"1981740 kB"}),
            dict(pid=12, role="engine", configured_allocator="mimalloc", observed_env_allocator=None,
                 observed_env_pythonmalloc=None, process_environment_reliable=False,
                 libraries=["/opt/mimalloc/lib/libmimalloc.so.2"], status={"VmRSS":"100 kB"}, memory={"Pss":"90 kB"})]
        result = analysis.process_memory(records, expected_allocator="mimalloc")
        self.assertEqual(result["memory_fields"]["vmrss_kb"]["sum"], 2557900)
        self.assertEqual(result["memory_fields"]["private_dirty_kb"]["sum"], 1981740)
        self.assertEqual(result["processes_with_mapped_mimalloc"], 2)
        self.assertEqual(result["activation"]["status"], "verified_for_captured_roles")
        self.assertEqual(result["activation"]["unreliable_engine_environment_records"], 1)
        records[0]["observed_env_pythonmalloc"] = "malloc"
        self.assertEqual(analysis.process_memory(records, expected_allocator="mimalloc")["activation"]["status"], "failed")
        records[0]["observed_env_pythonmalloc"] = "default"
        records[1]["libraries"] = []
        self.assertEqual(analysis.process_memory(records, expected_allocator="mimalloc")["activation"]["status"], "failed")

    def test_system_arm_requires_no_mimalloc_mapping_and_default_python_api(self):
        records = [dict(role=role, configured_allocator="system", observed_env_allocator="system",
            observed_env_pythonmalloc="default", process_environment_reliable=role == "api", libraries=[], status={"VmRSS":"100 kB"})
            for role in ("api", "engine")]
        self.assertEqual(analysis.process_memory(records, expected_allocator="system")["activation"]["status"], "verified_for_captured_roles")
        records[1]["libraries"] = ["/opt/libmimalloc.so"]
        self.assertEqual(analysis.process_memory(records, expected_allocator="system")["activation"]["status"], "failed")


if __name__ == "__main__": unittest.main()
