"""Synthetic current-correspondence tests: no Django, storage, or network."""
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "pdf_current_validation.py"
SPEC = importlib.util.spec_from_file_location("pdf_current_validation", MODULE_PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic(*texts, encrypted=False):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=200, height=200)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
                DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 10 100 Td <" + text.encode().hex().encode() + b"> Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("synthetic-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class CurrentValidationTests(unittest.TestCase):
    def test_unavailable_process_driver_does_not_start_a_child(self):
        raw, canonical = synthetic("example"), b"example"
        with patch.object(core.os, "name", "posix"), \
                patch.object(core, "Path", return_value=MODULE_PATH), \
                patch.object(core.importlib.util, "module_from_spec", side_effect=OSError), \
                patch.object(core.subprocess, "Popen") as spawn:
            result = core.validate(self.policy, raw, canonical, digest(canonical), time.monotonic() + 2)
        self.assertEqual(result["status"], "parsererror")
        spawn.assert_not_called()

    @classmethod
    def setUpClass(cls):
        cls.policy = core.freeze_policy()

    def evaluate(self, raw, canonical, stored=None, policy=None):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            return core._evaluate(policy or self.policy, raw, canonical,
                                  digest(canonical) if stored is None else stored)
        finally:
            logging.disable(previous)

    def test_exact_page_policy_preserves_internal_text(self):
        fixtures = [
            (["  alpha  ", " \n\t", None, " beta\n"], b"alpha\n\nbeta"),
            (["  a\t  b\n c  "], b"a\t  b\n c"),
            (["photometric pho-\ntometric \ufb01lter\x00"],
             "photometric pho-\ntometric \ufb01lter\x00".encode()),
            ([None, "\t\n"], b""),
        ]
        for texts, expected in fixtures:
            with self.subTest(texts=texts):
                reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda t=t: t) for t in texts])
                self.assertEqual(core._project(reader), (len(texts), expected))

    def test_real_pdf_match_omits_blank_pages_and_trailing_newline(self):
        result = self.evaluate(synthetic(" alpha ", "", "beta"), b"alpha\n\nbeta")
        self.assertEqual(result["status"], "match")
        self.assertEqual((result["pages"], result["projected_bytes"]), (3, 11))
        self.assertEqual(result["projected_sha256"], digest(b"alpha\n\nbeta"))
        self.assertNotIn("text", result)

    def test_final_newline_is_mismatch_instead_of_normalized(self):
        self.assertEqual(self.evaluate(synthetic("alpha"), b"alpha\n")["status"], "mismatch")

    def test_empty_projection_cannot_match_empty_canonical(self):
        self.assertEqual(self.evaluate(synthetic("", ""), b"")["status"], "empty")

    def test_hash_invalid_prevents_positive_result(self):
        for stored in ("0" * 64, digest(b"alpha").upper(), None, True):
            # None is passed directly to exercise strict digest types.
            self.assertEqual(core._evaluate(self.policy, synthetic("alpha"), b"alpha", stored)["status"], "hashinvalid")

    def test_distinct_failure_statuses(self):
        fixtures = [(synthetic("alpha", encrypted=True), b"alpha", "encrypted"),
                    (synthetic(*([""] * 101)), b"", "limits"),
                    (b"not a pdf", b"alpha", "parsererror"),
                    (b"x" * (32 * 1024 * 1024 + 1), b"alpha", "limits"),
                    (b"pdf", b"x" * (8 * 1024 * 1024 + 1), "limits"),
                    (synthetic("alpha"), b"\xff", "parsererror")]
        for raw, canonical, expected in fixtures:
            with self.subTest(expected=expected):
                self.assertEqual(self.evaluate(raw, canonical)["status"], expected)

    def test_extracted_projection_is_bounded_without_truncating(self):
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "x" * (8 * 1024 * 1024 + 1))])
        with self.assertRaises(core._ProjectionLimit):
            core._project(reader)

    def test_parser_memory_exhaustion_is_a_limit_failure(self):
        with patch.object(core, "_project", side_effect=MemoryError):
            self.assertEqual(self.evaluate(synthetic("alpha"), b"alpha")["status"], "limits")

    def test_policy_drift_is_rejected_before_parsing(self):
        for field in ("key", "version", "implementation_sha256", "runtime_sha256"):
            changed = dict(self.policy, **{field: "changed"})
            self.assertEqual(self.evaluate(b"not a pdf", b"alpha", policy=changed)["status"], "policychanged")
        changed = dict(self.policy, extra="untrusted")
        self.assertEqual(self.evaluate(b"not a pdf", b"alpha", policy=changed)["status"], "policychanged")

    def test_frozen_identity_binds_current_code_runtime_and_limits(self):
        self.assertEqual(set(self.policy), {"key", "version", "implementation_sha256", "runtime_sha256"})
        sources = [(name, digest(MODULE_PATH.with_name(name).read_bytes())) for name in
                   ("pdf_current_validation.py", "pdf_validation_process.py")]
        self.assertEqual(self.policy["implementation_sha256"], digest(core._json(sources)))
        for target, value in (("MAX_PAGES", 99), ("MAX_CANONICAL_BYTES", 1)):
            with patch.object(core, target, value):
                self.assertNotEqual(core.freeze_policy()["runtime_sha256"], self.policy["runtime_sha256"])
        with patch.object(sys, "version", "synthetic different Python"):
            self.assertNotEqual(core.freeze_policy()["runtime_sha256"], self.policy["runtime_sha256"])
        import pypdf
        with patch.object(pypdf, "__version__", "synthetic different pypdf"):
            self.assertNotEqual(core.freeze_policy()["runtime_sha256"], self.policy["runtime_sha256"])

    def test_dependency_source_change_changes_frozen_runtime(self):
        import pypdf
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "__init__.py"
            source.write_text("first synthetic implementation", encoding="utf-8")
            with patch.object(pypdf, "__file__", str(source)):
                first = core.freeze_policy()
                source.write_text("second synthetic implementation", encoding="utf-8")
                second = core.freeze_policy()
        self.assertNotEqual(first["runtime_sha256"], second["runtime_sha256"])

    def test_strict_child_receipts_cannot_forge_positive(self):
        base = core._base(b"pdf", b"alpha")
        good = dict(base, status="match", pages=1, projected_bytes=5,
                    projected_codepoints=5, projected_sha256=digest(b"alpha"))
        self.assertTrue(core._valid_receipt(good, base))
        variants = [dict(base, status="match"), dict(good, projected_bytes=True),
                    dict(good, pages=101), dict(good, pages=0), dict(good, pages=1.0),
                    dict(good, projected_codepoints=0), dict(good, projected_bytes=6),
                    dict(good, projected_sha256="0" * 64), dict(good, status="mismatch"),
                    dict(good, extra="private"), dict(good, status="parsererror"),
                    dict(good, canonical_bytes=True), dict(good, projected_bytes=8 * 1024 * 1024 + 1)]
        for receipt in variants:
            with self.subTest(receipt=receipt):
                self.assertFalse(core._valid_receipt(receipt, base))

    def test_parent_hash_limit_and_deadline_checks(self):
        raw = synthetic("alpha")
        self.assertEqual(core.validate(self.policy, raw, b"alpha", "0" * 64, time.monotonic() + 1)["status"], "hashinvalid")
        self.assertEqual(core.validate(self.policy, raw, b"alpha", digest(b"alpha"), time.monotonic() - 1)["status"], "timeout")
        for invalid in (True, float("inf"), float("nan")):
            self.assertEqual(core.validate(self.policy, raw, b"alpha", digest(b"alpha"), invalid)["status"], "parsererror")

    def test_mutable_inputs_are_not_captures(self):
        with self.assertRaises(TypeError):
            core.validate(self.policy, bytearray(b"pdf"), b"alpha", digest(b"alpha"), time.monotonic() + 1)

    @unittest.skipIf(os.name == "posix", "unsupported-platform check")
    def test_unsupported_platform_has_no_inprocess_fallback(self):
        result = core.validate(self.policy, synthetic("alpha"), b"alpha", digest(b"alpha"), time.monotonic() + 10)
        self.assertEqual(result["status"], "unsupportedplatform")


@unittest.skipUnless(os.name == "posix", "real POSIX worker tests run on the target")
class PosixWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = core.freeze_policy()

    def process(self, program):
        return subprocess.Popen([sys.executable, "-I", "-B", "-c", program],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, start_new_session=True)

    def test_real_isolated_match_and_drift(self):
        raw = synthetic("alpha", "", "beta")
        canonical = b"alpha\n\nbeta"
        result = core.validate(self.policy, raw, canonical, digest(canonical), time.monotonic() + 10)
        self.assertEqual(result["status"], "match")
        changed = dict(self.policy, runtime_sha256="0" * 64)
        result = core.validate(changed, raw, canonical, digest(canonical), time.monotonic() + 10)
        self.assertEqual(result["status"], "policychanged")

    def test_real_stalled_stdin_times_out_and_is_killed(self):
        process = self.process("import time; time.sleep(60)")
        status, output = core._pump(process, b"x" * (1024 * 1024), time.monotonic() + .1)
        self.assertEqual((status, output), ("timeout", b""))
        self.assertIsNotNone(process.poll())

    def test_real_native_fd_overflow_is_bounded_and_killed(self):
        process = self.process("import os,time; os.write(1,b'private'*1000000); time.sleep(60)")
        status, output = core._pump(process, b"", time.monotonic() + 2)
        self.assertEqual((status, output), ("limits", b""))
        self.assertIsNotNone(process.poll())

    def test_real_malformed_receipt_fails_closed(self):
        process = self.process("import sys; sys.stdin.buffer.read(); print('not-json')")
        with patch.object(core.subprocess, "Popen", return_value=process):
            result = core.validate(self.policy, synthetic("alpha"), b"alpha", digest(b"alpha"), time.monotonic() + 3)
        self.assertEqual(result["status"], "parsererror")

    def test_real_worker_suppresses_parser_native_and_python_output(self):
        raw = synthetic("alpha")
        request = core._request(self.policy, raw, b"alpha", digest(b"alpha"))
        program = (
            "import importlib.util,os,sys; "
            f"s=importlib.util.spec_from_file_location('tested', {str(MODULE_PATH)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "original=m._project; "
            "exec('def noisy(reader):\\n print(\"private stdout\")\\n os.write(1,b\"private native\")\\n os.write(2,b\"private stderr\")\\n return original(reader)'); "
            "m._project=noisy; m._worker()"
        )
        process = self.process(program)
        status, output = core._pump(process, request, time.monotonic() + 10)
        self.assertIsNone(status)
        self.assertNotIn(b"private", output)
        self.assertEqual(json.loads(output)["status"], "match")


if __name__ == "__main__":
    unittest.main()
