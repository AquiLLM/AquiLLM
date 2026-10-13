"""Pure host checks for isolated frontend test selection; no app imports."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rag_frontend_harness", ROOT / "scripts/run_rag_remediation_tests.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


class FrontendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (
            "react/src/One.test.ts",
            "react/src/components/Two.test.tsx",
            "react/src/components/Two.lifecycle.test.tsx",
            "react/src/Source.ts",
            "react/src/Pattern[1].test.ts",
            "outside.test.ts",
        ):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("export {};\n", encoding="utf-8")

    def resolve(self, paths):
        resolver = getattr(HARNESS, "resolve_vitest_specs", None)
        self.assertTrue(callable(resolver), "isolated Vitest selection is unavailable")
        return resolver(paths, root=self.root)

    def test_resolves_only_explicit_existing_component_or_pure_tests(self):
        self.assertEqual(
            self.resolve([
                "react/src/One.test.ts", "react/src/components/Two.test.tsx"
            ]),
            ("src/One.test.ts", "src/components/Two.test.tsx"),
        )

    def test_rejects_escape_glob_non_test_missing_and_option_inputs(self):
        for name in (
            "outside.test.ts", "react/../outside.test.ts",
            "react/src/Source.ts", "react/src/missing.test.ts",
            "react/src/*.test.ts", "react/src/Pattern[1].test.ts",
            "--config", "react/src", str(self.root.parent / "outside.test.ts"),
        ):
            with self.subTest(path=name):
                with self.assertRaises(HARNESS.IsolationError):
                    self.resolve([name])

    def test_resolves_existing_dotted_regression_test_names(self):
        self.assertEqual(self.resolve([
            "react/src/components/Two.lifecycle.test.tsx"
        ]), ("src/components/Two.lifecycle.test.tsx",))

    def test_rejects_empty_or_alias_duplicated_selection(self):
        for paths in ([], ["react/src/One.test.ts", "react/src/../src/One.test.ts"]):
            with self.subTest(paths=paths):
                with self.assertRaises(HARNESS.IsolationError):
                    self.resolve(paths)

    def test_new_mode_cannot_be_combined_with_other_commands(self):
        self.assertTrue(callable(getattr(HARNESS, "resolve_vitest_specs", None)))
        for extra in (
            ["--cleanup"], ["--playwright", "react/e2e/test.spec.ts"],
            ["--manage", "migrate"], ["--", "--config", "foreign.ts"],
        ):
            with self.subTest(extra=extra), patch.object(HARNESS, "ROOT", self.root):
                with patch.object(HARNESS, "docker", side_effect=AssertionError(
                    "invalid selection reached Docker"
                )):
                    self.assertEqual(HARNESS.main([
                        "--project", "aquillm-rag-unit", "--vitest",
                        "react/src/One.test.ts", *extra,
                    ]), 2)


if __name__ == "__main__":
    unittest.main()
