"""Fail-closed wrapper tests; runnable with stdlib before importing Django."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

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
        network = {"Name": project + "_isolated", "Internal": True, "Labels": labels}
        volume = {"Name": project + "_artifacts", "Labels": labels, "Driver": "local", "Options": None}
        container = {
            "Id": "abc", "Name": "/" + project + "-app-1",
            "Config": {"Labels": {**labels, "com.docker.compose.service": "app"},
                       "Env": [f"{k}={v}" for k, v in self.h.test_environment().items()]},
            "HostConfig": {"NetworkMode": project + "_isolated", "PortBindings": {}},
            "NetworkSettings": {"Networks": {project + "_isolated": {}}},
            "Mounts": [{"Type": "volume", "Name": volume["Name"]}],
        }
        self.h.validate_resources(project, [container], [network], [volume])
        mutations = (
            lambda c: c["Config"]["Labels"].pop(self.h.OWNER_LABEL),
            lambda c: c["NetworkSettings"]["Networks"].update({"bridge": {}}),
            lambda c: c["Mounts"].append({"Type": "bind", "Source": "/host/data"}),
            lambda c: c["Config"]["Env"].append("POSTGRES_HOST=production"),
            lambda c: c["HostConfig"].update({"PortBindings": {"8000/tcp": [{"HostPort": "8080"}]}}),
        )
        for change in mutations:
            unsafe = copy.deepcopy(container)
            change(unsafe)
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_resources(project, [unsafe], [network], [volume])
        for change in (lambda n: n.update(Internal=False), lambda n: n["Labels"].clear()):
            unsafe = copy.deepcopy(network)
            change(unsafe)
            with self.assertRaises(self.h.IsolationError):
                self.h.validate_resources(project, [container], [unsafe], [volume])

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


if __name__ == "__main__":
    unittest.main()
