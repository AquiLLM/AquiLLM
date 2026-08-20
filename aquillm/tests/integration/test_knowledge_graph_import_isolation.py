from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_disabled_graph_imports_do_not_load_optional_runtimes() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    blocked_modules = {
        "gliner2",
        "neo4j",
        "torch",
        "transformers",
        "peft",
        "huggingface_hub",
    }
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import os
        import sys

        BLOCKED = {blocked_modules!r}
        sys.path.insert(0, {str(repository_root / "aquillm")!r})

        class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition('.')[0] in BLOCKED:
                    raise AssertionError(
                        f'optional runtime import attempted: {{fullname}}'
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalRuntime())
        os.environ['KG_BUILD_ENABLED'] = '0'

        import lib.knowledge_graph.config
        import lib.knowledge_graph.extractors
        import lib.knowledge_graph.query_extractor.contracts
        import lib.knowledge_graph.retrieval_config
        import aquillm.settings
        import aquillm.asgi
        import apps.knowledge_graph.models
        import apps.knowledge_graph.projection.identifiers
        import apps.knowledge_graph.projection.records
        import apps.knowledge_graph.projection.serialization
        import apps.knowledge_graph.retrieval.branch_contracts
        import apps.knowledge_graph.retrieval.direct_seed_contracts
        import apps.knowledge_graph.retrieval.expansion
        import apps.knowledge_graph.retrieval.ppr
        import apps.knowledge_graph.retrieval.projected_types
        import apps.knowledge_graph.retrieval.topology.contracts
        from aquillm.celery import app

        app.autodiscover_tasks(['apps.knowledge_graph'], force=True)

        loaded = sorted(
            name for name in sys.modules if name.partition('.')[0] in BLOCKED
        )
        assert loaded == [], loaded
        """
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repository_root / "aquillm"),
            "DJANGO_DEBUG": "0",
            "SECRET_KEY": "test-only-secret-key",
            "GOOGLE_OAUTH2_CLIENT_ID": "test-client-id",
            "GOOGLE_OAUTH2_CLIENT_SECRET": "test-client-secret",
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GEMINI_API_KEY": "test-gemini-key",
            "KG_BUILD_ENABLED": "0",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
