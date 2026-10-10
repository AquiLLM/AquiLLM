"""Fail-closed wrapper tests; runnable with stdlib before importing Django."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
import tempfile
import sys
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/run_rag_remediation_tests.py"

if os.environ.get("RAG_REMEDIATION_TEST") == "1":
    import pytest
    pytestmark = pytest.mark.django_db


def harness():
    assert WRAPPER.exists(), "disposable RAG wrapper is required"
    spec = importlib.util.spec_from_file_location("rag_harness", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.h = harness()

    def test_reject_shared_project_names(self):
        for name in ("default", "compose", "aquillm", "production", "development", "test", "rag", "../other"):
            with self.subTest(name=name), self.assertRaises(self.h.IsolationError):
                self.h.validate_project(name)
        self.h.validate_project("aquillm-rag-review")

    def test_external_endpoint_rejected_without_startup_or_network(self):
        env = self.h.test_environment()
        for key, value in (
            ("POSTGRES_HOST", "production-db"),
            ("POSTGRES_NAME", "production"),
            ("DJANGO_CACHE_REDIS_URL", "redis://cache.example:6379/0"),
            ("APP_EMBED_BASE_URL", "https://api.openai.com/v1"),
            ("OPENAI_BASE_URL", "http://localhost:8080/v1"),
            ("KG_MEMGRAPH_URI", "bolt://shared:7687"),
            ("KG_PROJECTION_POSTGRES_SOURCE_DSN", "postgresql://user:password@foreign/db"),
            ("UNEXPECTED_MODEL_ENDPOINT", "https://foreign.example"),
        ):
            with self.subTest(key=key), self.assertRaises(self.h.IsolationError):
                self.h.validate_environment({**env, key: value})
        self.h.validate_environment(env)

    def test_validates_owned_containers_networks_and_volumes(self):
        project = "aquillm-rag-review"
        labels = {"com.docker.compose.project": project, self.h.OWNER_LABEL: "v1"}
        network = {"Name": project + "_isolated", "Internal": True, "Driver": "bridge", "Labels": labels}
        volume = {"Name": project + "_artifacts", "Labels": labels, "Driver": "local", "Options": None}
        container = {
            "Id": "abc", "Name": "/" + project + "-app-1",
            "Config": {"Labels": {**labels, "com.docker.compose.service": "app"},
                       "Env": [f"{k}={v}" for k, v in self.h.test_environment().items()]},
            "HostConfig": {"NetworkMode": project + "_isolated", "PortBindings": {}},
            "NetworkSettings": {"Networks": {project + "_isolated": {"Aliases": ["app"]}}},
            "Mounts": [{"Type": "volume", "Name": volume["Name"]}],
        }
        peers = [container]
        for service in sorted(self.h.SERVICES - {"app", "frontend"}):
            peer = copy.deepcopy(container)
            peer["Id"] = service + "-id"
            peer["Config"]["Labels"]["com.docker.compose.service"] = service
            peer["NetworkSettings"]["Networks"][project + "_isolated"]["Aliases"] = [service]
            peers.append(peer)
        network["Containers"] = {c["Id"]: {} for c in peers}
        self.h.validate_resources(project, peers, [network], [volume])
        mutations = (
            lambda c: c["Config"]["Labels"].pop(self.h.OWNER_LABEL),
            lambda c: c["NetworkSettings"]["Networks"].update({"bridge": {}}),
            lambda c: c["Mounts"].append({"Type": "bind", "Source": "/host/data"}),
            lambda c: c["Config"]["Env"].append("POSTGRES_HOST=production"),
            lambda c: c["HostConfig"].update({"PortBindings": {"8000/tcp": [{"HostPort": "8080"}]}}),
            lambda c: c["HostConfig"].update({"CapAdd": ["NET_ADMIN"]}),
            lambda c: c["HostConfig"].update({"PidMode": "host"}),
            lambda c: c["NetworkSettings"]["Networks"][project + "_isolated"].update(Aliases=["db"]),
        )
        for change in mutations:
            unsafe = copy.deepcopy(container)
            change(unsafe)
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_resources(project, [unsafe, *peers[1:]], [network], [volume])
        for change in (lambda n: n.update(Internal=False), lambda n: n["Labels"].clear()):
            unsafe = copy.deepcopy(network)
            change(unsafe)
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_resources(project, peers, [unsafe], [volume])

    def test_compose_foreign_network_rejected_before_creation(self):
        project = "aquillm-rag-review"
        config = {
            "services": {name: {"labels": {self.h.OWNER_LABEL: "v1"}, "networks": {"isolated": {}}, "environment": self.h.test_environment() if name == "app" else {}} for name in self.h.SERVICES},
            "networks": {"isolated": {"name": project + "_isolated", "internal": True, "labels": {self.h.OWNER_LABEL: "v1"}}},
            "volumes": {},
        }
        for name, service in config["services"].items():
            if name in {"app", "fake-models", "frontend"}:
                service["build"] = {"context": str(ROOT), "dockerfile": "deploy/docker/test/Dockerfile.rag-remediation-browser" if name == "frontend" else "deploy/docker/test/Dockerfile.rag-remediation"}
            else:
                service["image"] = {"db": "pgvector/pgvector:pg16", "redis": "redis:7.4-bookworm", "memgraph": "memgraph/memgraph-mage:3.8.1"}[name]
        self.h.validate_compose(config, project)
        for change in (lambda n: n.update(external=True), lambda n: n.update(name="shared"), lambda n: n["labels"].clear()):
            unsafe = copy.deepcopy(config)
            change(unsafe["networks"]["isolated"])
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_compose(unsafe, project)
        for field, value in (("context", str(ROOT.parent)), ("dockerfile", "../foreign.Dockerfile")):
            unsafe = copy.deepcopy(config)
            unsafe["services"]["app"]["build"][field] = value
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_compose(unsafe, project)

    def test_cli_refusal_occurs_before_any_process_launch(self):
        for key, value in (("POSTGRES_HOST", "production"), ("DJANGO_CACHE_REDIS_URL", "redis://foreign:6379"), ("APP_EMBED_BASE_URL", "https://foreign.example/private-key")):
            errors = StringIO()
            with patch.dict(os.environ, {key: value}), patch.object(self.h.subprocess, "run", side_effect=AssertionError("process launched")), redirect_stderr(errors):
                self.assertEqual(self.h.main(["--project", "aquillm-rag-review", "--", "unused.py"]), 2)
            self.assertIn(key, errors.getvalue())
            self.assertNotIn(value, errors.getvalue())

    def test_cli_rejects_settings_and_configuration_overrides_before_docker(self):
        for tokens in (("--ds", "aquillm.settings"), ("--ds=aquillm.settings",), ("-c", "foreign.ini"), ("-cforeign.ini",), ("--override-ini=DJANGO_SETTINGS_MODULE=aquillm.settings",), ("-o", "DJANGO_SETTINGS_MODULE=aquillm.settings"), ("--rootdir=foreign",)):
            with self.subTest(tokens=tokens), patch.object(self.h.subprocess, "run", side_effect=AssertionError("Docker launched")), redirect_stderr(StringIO()):
                self.assertEqual(self.h.main(["--project", "aquillm-rag-review", "--", *tokens]), 2)
        for tokens in (("showmigrations", "--settings", "aquillm.settings"), ("showmigrations", "--settings=aquillm.settings"), ("check", "--pythonpath=foreign"), ("check", "--set", "aquillm.settings"), ("check", "--set=aquillm.settings"), ("check", "--py", "foreign"), ("check", "--py=foreign")):
            with self.subTest(tokens=tokens), patch.object(self.h.subprocess, "run", side_effect=AssertionError("Docker launched")), redirect_stderr(StringIO()):
                self.assertEqual(self.h.main(["--project", "aquillm-rag-review", "--manage", *tokens]), 2)

    def test_actual_django_parser_abbreviations_cannot_bypass_guard(self):
        if os.environ.get("RAG_REMEDIATION_TEST") != "1":
            self.skipTest("real Django CommandParser assertion requires installed apps")
        from django.core.management import ManagementUtility
        command = ManagementUtility().fetch_command("check")
        parser = command.create_parser("manage.py", "check")
        for tokens in (("--set", "aquillm.settings"), ("--set=aquillm.settings",), ("--py", "foreign"), ("--py=foreign",)):
            parsed = parser.parse_args(tokens)
            self.assertTrue(parsed.settings == "aquillm.settings" or parsed.pythonpath == "foreign")
            with patch.object(self.h.subprocess, "run", side_effect=AssertionError("Docker launched")), redirect_stderr(StringIO()):
                self.assertEqual(self.h.main(["--project", "aquillm-rag-review", "--manage", "check", *tokens]), 2)

    def test_incomplete_container_inventory_and_unowned_peers_rejected(self):
        project = "aquillm-rag-review"
        labels = {"com.docker.compose.project": project, self.h.OWNER_LABEL: "v1"}
        app = {"Id": "app-id", "Config": {"Labels": {**labels, "com.docker.compose.service": "app"}, "Env": [f"{k}={v}" for k, v in self.h.test_environment().items()]}, "HostConfig": {"NetworkMode": project + "_isolated"}, "NetworkSettings": {"Networks": {project + "_isolated": {"Aliases": ["app"]}}}, "Mounts": []}
        network = {"Name": project + "_isolated", "Internal": True, "Driver": "bridge", "Labels": labels, "Containers": {"app-id": {}}}
        with self.assertRaises(self.h.IsolationError):
            self.h.validate_resources(project, [app], [network], [])
        peers = []
        for service in self.h.SERVICES - {"frontend"}:
            peer = copy.deepcopy(app)
            peer["Id"] = service + "-id"
            peer["Config"]["Labels"]["com.docker.compose.service"] = service
            peer["NetworkSettings"]["Networks"][project + "_isolated"]["Aliases"] = [service]
            peers.append(peer)
        network["Containers"] = {c["Id"]: {} for c in peers}
        self.h.validate_resources(project, peers, [network], [])
        network["Containers"]["unowned-peer"] = {}
        with self.assertRaises(self.h.IsolationError):
            self.h.validate_resources(project, peers, [network], [])

    def test_records_unicode_output_and_failure_artifacts(self):
        self.assertTrue(hasattr(self.h, "run_recorded"), "runner must retain test output and handle Windows Unicode")
        with tempfile.TemporaryDirectory() as temp:
            output = StringIO()
            with patch.object(sys, "stdout", output):
                result = self.h.run_recorded([sys.executable, "-c", "print(chr(10003)); raise SystemExit(1)"], Path(temp), "aquillm-rag-review")
            self.assertEqual(result, 1)
            self.assertIn("\u2713", output.getvalue())
            self.assertIn("\u2713", (Path(temp) / "test-output.log").read_text(encoding="utf-8"))
            self.assertIn('"exit_code": 1', (Path(temp) / "result.json").read_text())

    def test_foreign_browser_url_storage_and_spec_rejected(self):
        self.h.validate_browser("http://app:8000", "/artifacts/browser/run-123/state.json")
        for url, storage in (
            ("http://localhost:8080", "/artifacts/browser/run-123/state.json"),
            ("https://foreign.example", "/artifacts/browser/run-123/state.json"),
            ("http://app:8000", "auth.json"),
            ("http://app:8000", "/artifacts/browser/../../auth.json"),
        ):
            with self.subTest(url=url, storage=storage), self.assertRaises(self.h.IsolationError):
                self.h.validate_browser(url, storage)
        for spec in ("../outside.spec.ts", "aquillm/manage.py", "react/../auth.json"):
            with self.assertRaises(self.h.IsolationError):
                self.h.resolve_spec(spec)

    def test_full_registry_and_live_postgres_memgraph(self):
        if os.environ.get("RAG_REMEDIATION_TEST") != "1":
            self.skipTest("live adapter assertion runs only inside owned Compose app")
        from django.apps import apps
        from django.db import connection
        from django.conf import settings
        from neo4j import GraphDatabase
        self.assertEqual(connection.vendor, "postgresql")
        self.assertTrue(apps.is_installed("apps.knowledge_graph"))
        self.assertTrue(apps.is_installed("apps.documents"))
        with connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            self.assertIn("PostgreSQL 16", cursor.fetchone()[0])
        with GraphDatabase.driver(settings.KG_MEMGRAPH_URI, auth=None) as driver:
            with driver.session() as session:
                self.assertEqual(session.run("RETURN 1 AS ready").single()["ready"], 1)

    def test_owned_frontend_bundle_uses_application_static_prefix(self):
        if os.environ.get("RAG_REMEDIATION_TEST") != "1" or not Path("/frontend-build/main.js").exists():
            self.skipTest("frontend assertion requires the owned browser build volume")
        from django.contrib.staticfiles import finders
        self.assertEqual(finders.find("js/dist/main.js"), "/frontend-build/main.js")

    def test_owned_fake_providers_are_deterministic_and_match_dimensions(self):
        if os.environ.get("RAG_REMEDIATION_TEST") != "1":
            self.skipTest("HTTP contract requires the owned fake-model service")
        def post(path, body):
            request = Request("http://fake-models:8000" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                return json.load(response)
        first = post("/v1/embeddings", {"input": ["synthetic source"], "dimensions": 1024})
        second = post("/v1/embeddings", {"input": ["synthetic source"], "dimensions": 1024})
        self.assertEqual(first, second)
        self.assertEqual(len(first["data"][0]["embedding"]), 1024)
        scores = post("/v1/rerank", {"query": "synthetic source", "documents": ["synthetic source", "other"]})
        self.assertGreater(scores["results"][0]["relevance_score"], scores["results"][1]["relevance_score"])
        with self.assertRaises(HTTPError) as error:
            post("/unsupported", {})
        self.assertEqual(error.exception.code, 501)


if __name__ == "__main__":
    unittest.main()
