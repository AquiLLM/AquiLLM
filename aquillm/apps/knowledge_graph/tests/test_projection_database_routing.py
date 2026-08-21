from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from apps.knowledge_graph.models import (
    CollectionGraphProjection,
    GraphArtifact,
    ProjectionChunkReference,
)
from aquillm.projection_database_settings import projection_databases


def test_projection_router_is_narrow_and_never_migrates_worker_aliases() -> None:
    from apps.knowledge_graph.projection.database_router import ProjectionDatabaseRouter

    router = ProjectionDatabaseRouter()
    assert (
        router.db_for_read(GraphArtifact, projection_source=True) == "projection_source"
    )
    assert router.db_for_read(GraphArtifact) is None
    assert router.db_for_write(CollectionGraphProjection) is None
    assert (
        router.db_for_write(CollectionGraphProjection, projection_worker_state=True)
        == "projection_state"
    )
    assert router.db_for_write(GraphArtifact, projection_worker_state=True) is None
    assert router.allow_migrate("projection_source", "apps_knowledge_graph") is False
    assert router.allow_migrate("projection_state", "apps_knowledge_graph") is False
    assert router.allow_migrate("default", "apps_knowledge_graph") is None


def test_projection_settings_define_separate_fail_closed_database_aliases() -> None:
    settings_root = Path(__file__).resolve().parents[3] / "aquillm"
    settings_source = (settings_root / "settings.py").read_text(encoding="utf-8")
    alias_source = (settings_root / "projection_database_settings.py").read_text(
        encoding="utf-8"
    )
    assert "projection_databases()" in settings_source
    assert "ProjectionDatabaseRouter" in settings_source
    for expected in (
        "KG_PROJECTION_POSTGRES_SOURCE_DSN",
        "KG_PROJECTION_POSTGRES_STATE_DSN",
        '"projection_source"',
        '"projection_state"',
        "urlsplit",
        "unquote",
    ):
        assert expected in alias_source


def test_projection_aliases_accept_frozen_passwordless_default_port_dsns() -> None:
    databases = projection_databases(
        {
            "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
            "KG_PROJECTION_POSTGRES_SOURCE_DSN": (
                "postgresql://aquillm_projection_source@pg.internal/aquillm"
            ),
            "KG_PROJECTION_POSTGRES_STATE_DSN": (
                "postgresql://aquillm_projection_state@pg.internal/aquillm"
            ),
        }
    )
    assert databases["projection_source"] == {
        "ENGINE": "django_prometheus.db.backends.postgresql",
        "NAME": "aquillm",
        "USER": "aquillm_projection_source",
        "PASSWORD": "",
        "HOST": "pg.internal",
        "PORT": "5432",
        "OPTIONS": {"connect_timeout": 5},
    }
    assert databases["projection_state"]["USER"] == "aquillm_projection_state"


@pytest.mark.parametrize(("key", "username"), (
    ("KG_PROJECTION_POSTGRES_SOURCE_DSN", "unexpected_source"),
    ("KG_PROJECTION_POSTGRES_STATE_DSN", "unexpected_state"),
))
def test_projection_aliases_reject_non_authority_roles(
    key: str, username: str
) -> None:
    source = {
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": (
            "postgresql://aquillm_projection_source@pg/aquillm"
        ),
        "KG_PROJECTION_POSTGRES_STATE_DSN": (
            "postgresql://aquillm_projection_state@pg/aquillm"
        ),
    }
    source[key] = f"postgresql://{username}@pg/aquillm"

    with pytest.raises(ValueError, match=key):
        projection_databases(source)


def test_production_alias_contract_resolves_in_django_settings() -> None:
    from django.conf import settings

    from apps.knowledge_graph.projection.runtime import ProjectionDatabaseAliases

    aliases = ProjectionDatabaseAliases()
    assert aliases.source == "projection_source"
    assert aliases.state == "projection_state"
    assert aliases.source in settings.DATABASES
    assert aliases.state in settings.DATABASES


@pytest.mark.parametrize("module_name", ("generation_audit.py", "inspection.py"))
def test_projection_authority_reads_never_use_function_only_state_alias(
    module_name: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "projection" / module_name
    ).read_text(encoding="utf-8")
    assert "ProjectionDatabaseAliases().source" in source
    assert "ProjectionDatabaseAliases().state" not in source


def test_worker_state_repository_calls_only_fixed_function_names(monkeypatch) -> None:
    from apps.knowledge_graph.projection import state_repository

    calls: list[tuple[str, tuple[object, ...]]] = []

    class Cursor:
        description = (
            ("projection_id",),
            ("owner",),
            ("expires_at",),
            ("attempt_count",),
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql, params):
            calls.append((sql, tuple(params)))

        def fetchone(self):
            return None

    class Connection:
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(
        state_repository, "connections", {"projection_state": Connection()}
    )
    repository = state_repository.FunctionProjectionStateRepository(
        state_using="projection_state", source_using="projection_source"
    )
    repository.claim(
        projection_id=state_repository.UUID("00000000-0000-0000-0000-000000000001"),
        owner="worker-1",
        now=state_repository.datetime(2026, 8, 20, tzinfo=state_repository.UTC),
        lease_seconds=60,
    )
    assert calls == [
        (
            "SELECT * FROM public.kg_projection_claim(%s, %s, %s, %s)",
            (
                state_repository.UUID("00000000-0000-0000-0000-000000000001"),
                "worker-1",
                state_repository.datetime(2026, 8, 20, tzinfo=state_repository.UTC),
                60,
            ),
        )
    ]
    source = inspect.getsource(state_repository.FunctionProjectionStateRepository)
    assert '.using("projection_state")' not in source
    assert "UPDATE " not in source
    assert "INSERT " not in source
    assert "DELETE " not in source


def test_worker_and_outbox_use_function_repository_but_web_stays_default() -> None:
    root = Path(__file__).resolve().parents[1] / "projection"
    worker = (root / "worker.py").read_text(encoding="utf-8")
    outbox = (root / "outbox.py").read_text(encoding="utf-8")
    runtime = (root / "runtime.py").read_text(encoding="utf-8")
    assert "FunctionProjectionStateRepository" in worker
    assert "state_repository=" in worker
    assert "FunctionProjectionStateRepository" in outbox
    assert 'using == "projection_state"' in outbox
    assert '.using("projection_state")' not in outbox
    assert 'using="default"' in runtime


def test_0008_owns_a_function_only_projection_state_api() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0008_projection_worker_state_api"
    )
    sql = migration.STATE_API_SQL
    function_names = {
        "kg_projection_claim",
        "kg_projection_renew",
        "kg_projection_fail",
        "kg_projection_supersede",
        "kg_projection_store_chunk_references",
        "kg_projection_fence_chunk_references",
        "kg_projection_claim_outbox",
        "kg_projection_complete_outbox",
        "kg_projection_fail_outbox",
        "kg_projection_ready_compare_and_set",
    }
    for name in function_names:
        assert f"FUNCTION public.{name}" in sql
        assert f"REVOKE ALL ON FUNCTION public.{name}" in sql
    assert sql.count("SECURITY DEFINER") >= len(function_names)
    assert sql.count("SET search_path = pg_catalog, public") >= len(function_names)
    assert "GRANT EXECUTE ON FUNCTION" in sql
    assert "aquillm_projection_state" in sql
    state_grants = "\n".join(
        line for line in sql.splitlines() if "aquillm_projection_state" in line
    )
    assert "GRANT SELECT" not in state_grants
    assert "GRANT INSERT" not in state_grants
    assert "GRANT UPDATE" not in sql
    assert "GRANT DELETE" not in sql
    assert "GRANT ALL" not in sql


def test_0008_uses_real_collection_and_chunk_tables_and_requires_roles() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0008_projection_worker_state_api"
    )
    sql = migration.STATE_API_SQL

    assert "public.aquillm_collection" in sql
    assert "public.aquillm_textchunk" in sql
    assert "apps_collections_collection" not in sql
    assert "apps_documents_textchunk" not in sql
    for role in ("aquillm_projection_source", "aquillm_projection_state"):
        assert f"required role {role} is missing" in sql
    assert "IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles" not in sql


def test_ready_cas_has_exact_predicates_and_lock_order() -> None:
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0008_projection_worker_state_api"
    )
    sql = migration.STATE_API_SQL
    ready = sql.split("FUNCTION public.kg_projection_ready_compare_and_set", 1)[1]
    ready = ready.split("BEGIN", 1)[1]
    assert ready.index("aquillm_collection") < ready.index("graphartifact")
    assert ready.index("graphartifact") < ready.index("collectiongraphmembershipstate")
    assert ready.index("collectiongraphmembershipstate") < ready.index(
        "collectiongraphprojection"
    )
    for predicate in (
        "generation_key",
        "collection_id",
        "artifact_id",
        "schema_version",
        "projection_version",
        "identifier_key_version",
        "membership_checksum",
        "private_mapping_checksum",
        "validation_checksum",
        "FOR UPDATE",
    ):
        assert predicate in ready


@pytest.mark.parametrize(
    "forbidden",
    (
        "cursor.execute(sql",
        "cursor.execute(query",
        "raw(",
        "executescript",
        'projection_state").update(',
        'projection_state").create(',
        'projection_state").delete(',
    ),
)
def test_function_api_exposes_no_arbitrary_sql_or_direct_dml(forbidden: str) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "projection" / "state_repository.py"
    ).read_text(encoding="utf-8")
    assert forbidden not in source


def test_projection_source_owns_read_only_chunk_and_projection_reads() -> None:
    from apps.knowledge_graph.projection.state_repository import (
        FunctionProjectionStateRepository,
    )

    repository = FunctionProjectionStateRepository()
    assert repository.source_using == "projection_source"
    assert repository.state_using == "projection_state"
    assert repository.model_read_alias(CollectionGraphProjection) == "projection_source"
    assert repository.model_read_alias(ProjectionChunkReference) == "projection_source"
