"""Pure host-stdlib command admission checks; no Docker or application execution."""
import importlib.util
import json
import os
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def harness():
    spec = importlib.util.spec_from_file_location(
        "rag_typecheck_harness", ROOT / "scripts/run_rag_remediation_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TypecheckAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.h = harness()
        self.project = "aquillm-rag-review"

    def invoke(self, arguments):
        try:
            return self.h.main(["--project", self.project, *arguments])
        except SystemExit as error:
            return error.code

    def test_mixed_modes_and_extra_command_cannot_launch(self):
        alternatives = (
            ["--cleanup"], ["--playwright", "react/tests/usersettings.spec.ts"],
            ["--vitest", "react/src/utils/textOffsetMatch.test.ts"],
            ["--manage", "check"], ["--", "npm", "run", "other"],
            ["other"], ["--typecheck=other"],
        )
        for tokens in alternatives:
            with self.subTest(tokens=tokens), redirect_stderr(StringIO()), \
                    patch.object(self.h, "docker", side_effect=AssertionError("Docker launched")), \
                    patch.object(self.h.subprocess, "run", side_effect=AssertionError("process launched")):
                self.assertEqual(self.invoke(["--typecheck", *tokens]), 2)

    def test_inherited_endpoint_and_foreign_browser_refused(self):
        cases = (({"POSTGRES_HOST": "foreign"}, []),
                 ({}, ["--base-url", "https://foreign.invalid"]),
                 ({}, ["--storage-state", "/artifacts/browser/run-123/state.json"]))
        for env, tokens in cases:
            with self.subTest(tokens=tokens, env=env), patch.dict(os.environ, env, clear=True), \
                    redirect_stderr(StringIO()), \
                    patch.object(self.h, "docker", side_effect=AssertionError("Docker launched")):
                self.assertEqual(self.invoke(["--typecheck", *tokens]), 2)

    def test_resource_validation_refusal_prevents_dispatch(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(StringIO()), \
                patch.object(self.h, "inspect_project", side_effect=self.h.IsolationError("unowned")) as inspect, \
                patch.object(self.h, "docker") as docker, patch.object(self.h, "run_recorded") as record:
            self.assertEqual(self.invoke(["--typecheck"]), 2)
            inspect.assert_called_once_with(self.project, require_complete=False)
            docker.assert_not_called()
            record.assert_not_called()

    def run_stubbed_frontend(self, *, workdir="/app/react", count=1, result=0):
        app = {"Id": "app-id", "Name": "/owned-app", "Config": {
            "Labels": {"com.docker.compose.service": "app"}}}
        frontend = {"Id": "frontend-id", "Name": "/owned-frontend", "Config": {
            "Labels": {"com.docker.compose.service": "frontend"}, "WorkingDir": workdir}}
        resources = ([app, *([frontend] * count)], [], [])
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
            stack.enter_context(patch.object(self.h, "ROOT", Path(temp)))
            inspect = stack.enter_context(patch.object(self.h, "inspect_project", return_value=resources))
            docker = stack.enter_context(patch.object(self.h, "docker", return_value="{}"))
            validate = stack.enter_context(patch.object(self.h, "validate_compose"))
            stack.enter_context(patch.object(self.h, "prepare_git_provenance", return_value="a" * 40))
            record = stack.enter_context(patch.object(self.h, "run_recorded", return_value=result))
            code = self.invoke(["--typecheck"])
            return code, inspect.call_args_list, docker.call_args_list, validate.call_args_list, record.call_args_list

    def test_fixed_typecheck_build_validation_and_failure_exit(self):
        for expected in (0, 17):
            with self.subTest(expected=expected):
                code, inspect, docker, validate, record = self.run_stubbed_frontend(result=expected)
                self.assertEqual(code, expected)
                self.assertEqual(len(inspect), 3)
                self.assertEqual(inspect[0].kwargs, {"require_complete": False})
                self.assertEqual(len(validate), 1)
                self.assertTrue(any(c.args[-6:] == ("--profile", "browser", "up", "--build", "-d", "frontend") for c in docker))
                self.assertEqual(len(record), 1)
                command, artifact, project = record[0].args
                self.assertEqual(command[-6:], ["exec", "-T", "frontend", "npm", "run", "typecheck"])
                self.assertEqual(project, self.project)
                self.assertIn("run-", artifact.name)
                self.assertNotIn("--", command)

    def test_wrong_or_missing_frontend_cannot_dispatch(self):
        for options in ({"workdir": "/foreign"}, {"count": 0}, {"count": 2}):
            with self.subTest(options=options):
                code, _, _, _, record = self.run_stubbed_frontend(**options)
                self.assertEqual(code, 2)
                self.assertEqual(record, [])


if __name__ == "__main__":
    unittest.main()
