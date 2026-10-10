import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("allocator_analysis", Path(__file__).resolve().parents[1] / "allocator_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

FROZEN_INPUTS = {
    512: "ccf4d444c1d900d4281b8f07d34c5cd6e587be888741482785e84eb38cc15bf7",
    8192: "88ae468ccd6729be136d4191ebf675bf87cf6d7c876ded49644f2ea646ead7da",
    32768: "25926b18263a0cd6dec926b21d7cbbf8f73e0b16ff6c7f156ebf5013de03c38b",
    36864: "f099955b32a3c27bde10f3e7a27dad4cebd8e08de02a5f727661db0cf70bdd49",
}


def write_captures(directory, transform=lambda rows: rows):
    for role in ("system", "mimalloc"):
        for block in range(1, 4):
            rows = [dict(label=f"{role}-block{block}", input_sha256=digest, prompt_tokens=length,
                requested_output_tokens=256, output_tokens=256, output_sha256="b" * 64,
                usage={"completion_tokens": 256}, repeat=repeat, warmup=repeat < 0,
                complete=True, error=None, stream_done=True, finish_reason="length",
                ttft_seconds=1., total_seconds=2., aggregate_decode_seconds_per_token=1/255)
                for length, digest in FROZEN_INPUTS.items() for repeat in (-1, 0, 1)]
            (directory / f"h100-allocator-{role}-block{block}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in transform(rows)))


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

    def test_complete_frozen_shape_and_repeat_coverage_can_pass_request_gate(self):
        with tempfile.TemporaryDirectory() as name:
            write_captures(Path(name))
            result = analysis.build(Path(name), 2)
        self.assertTrue(result["protocol_screen"]["request_evidence_complete"])

    def test_empty_files_cannot_pass_request_gate(self):
        with tempfile.TemporaryDirectory() as name:
            write_captures(Path(name), lambda rows: [])
            result = analysis.build(Path(name), 2)
        self.assertFalse(result["protocol_screen"]["request_evidence_complete"])
        self.assertTrue(result["serving_errors"])

    def test_same_whole_shape_missing_in_every_arm_cannot_pass_request_gate(self):
        with tempfile.TemporaryDirectory() as name:
            write_captures(Path(name), lambda rows: [row for row in rows if row["prompt_tokens"] != 36864])
            result = analysis.build(Path(name), 2)
        self.assertFalse(result["protocol_screen"]["request_evidence_complete"])
        self.assertTrue(result["serving_errors"])

    def test_wrong_frozen_input_or_output_size_cannot_pass_even_when_arms_match(self):
        for change in ("hash", "output"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                def transform(rows):
                    for row in rows:
                        if change == "hash": row["input_sha256"] = "a" * 64
                        else: row.update(requested_output_tokens=128, output_tokens=128, usage={"completion_tokens":128})
                    return rows
                write_captures(Path(name), transform)
                result = analysis.build(Path(name), 2)
                self.assertFalse(result["protocol_screen"]["request_evidence_complete"])

    def test_missing_duplicate_or_malformed_warmup_cannot_pass_request_gate(self):
        for change in ("missing", "duplicate", "marker", "failed", "label"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                def transform(rows):
                    if change == "missing": return [row for row in rows if not row["warmup"]]
                    if change == "duplicate": return rows + [dict(rows[0])]
                    if change == "marker": rows[0]["warmup"] = False
                    if change == "failed": rows[0].update(complete=False, error="missing_done", stream_done=False)
                    if change == "label": rows[0]["label"] = "wrong-arm"
                    return rows
                write_captures(Path(name), transform)
                result = analysis.build(Path(name), 2)
                self.assertFalse(result["protocol_screen"]["request_evidence_complete"])

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

    def test_competing_or_unexpected_allocator_libraries_cannot_verify_activation(self):
        for mode, libraries in (("system", ["/opt/jemalloc/lib/libjemalloc.so"]),
                ("mimalloc", ["/opt/mimalloc/lib/libmimalloc.so.2", "/opt/jemalloc/lib/libjemalloc.so"]),
                ("mimalloc", ["/external/libmimalloc.so"]), ("mimalloc", ["/opt/mimalloc/../libmimalloc.so"])):
            with self.subTest(mode=mode, libraries=libraries):
                records = [dict(role=role, configured_allocator=mode, observed_env_allocator=mode,
                    observed_env_pythonmalloc="default", process_environment_reliable=role == "api",
                    libraries=libraries, status={"VmRSS":"100 kB"}) for role in ("api", "engine")]
                self.assertEqual(analysis.process_memory(records, expected_allocator=mode)["activation"]["status"], "failed")

    def test_unreliable_engine_environment_rejects_explicit_contrary_values(self):
        for name, value in (("observed_env_allocator", "system"), ("observed_env_pythonmalloc", "malloc")):
            with self.subTest(name=name):
                records = [dict(role=role, configured_allocator="mimalloc", observed_env_allocator="mimalloc",
                    observed_env_pythonmalloc="default", process_environment_reliable=role == "api",
                    libraries=["/opt/mimalloc/lib/libmimalloc.so.2"], status={"VmRSS":"100 kB"}) for role in ("api", "engine")]
                records[1][name] = value
                self.assertEqual(analysis.process_memory(records, expected_allocator="mimalloc")["activation"]["status"], "failed")


if __name__ == "__main__": unittest.main()
