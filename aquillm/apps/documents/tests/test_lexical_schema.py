"""Real PostgreSQL catalog installation and guarded migration reversal."""

from importlib import import_module

import pytest
from django.apps import apps
from django.db import connection

from apps.documents.models import TextChunk
from apps.documents.services.lexical_index import index_ready
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_lexical_publication import publish

pytestmark = pytest.mark.django_db(transaction=True)


def test_index_reverse_preserves_provenance_and_reinstall_is_idempotent(legacy):
    migration = import_module(
        "apps.documents.migrations.0012_chunk_search_projection_gin_v1"
    )
    publish(legacy)
    binding = TextChunk.objects.get().search_vector_binding
    assert index_ready("default")
    try:
        with connection.schema_editor(atomic=False) as editor:
            migration.drop_index(apps, editor)
        assert not index_ready("default")
        assert TextChunk.objects.get().search_vector_binding == binding
        from apps.documents.services.chunk_search_scope import freeze_search_scope
        from apps.documents.services.lexical_readiness import full_text_candidates

        scope = freeze_search_scope((legacy[0],), using="default")
        rows, coverage = full_text_candidates(TextChunk, "source", scope, limit=12)
        assert rows == ()
        assert coverage.status == "index_unavailable"
    finally:
        with connection.schema_editor(atomic=False) as editor:
            migration.create_index(apps, editor)
            migration.create_index(apps, editor)
    assert index_ready("default")


@pytest.mark.parametrize("flag", ["indisvalid", "indisready"])
def test_same_shape_incomplete_catalog_attempt_is_rebuilt(legacy, flag):
    """Simulate each interrupted-build catalog flag in the disposable test DB."""
    migration = import_module(
        "apps.documents.migrations.0012_chunk_search_projection_gin_v1"
    )
    try:
        with connection.cursor() as cursor:
            # flag is a closed test-owned enum, never external input.
            cursor.execute(
                f"UPDATE pg_index SET {flag}=false "
                "WHERE indexrelid='textchunk_search_vector_gin_v1'::regclass"
            )
        assert not index_ready("default")
        with connection.schema_editor(atomic=False) as editor:
            migration.create_index(apps, editor)
        assert index_ready("default")
    finally:
        with connection.schema_editor(atomic=False) as editor:
            migration.create_index(apps, editor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_kind", "table"),
        ("section_path", ["Measured"]),
        ("search_vector_binding", {}),
    ],
)
def test_schema_reverse_refuses_used_metadata(legacy, field, value):
    migration = import_module(
        "apps.documents.migrations.0011_chunk_search_projection_v1"
    )
    TextChunk.objects.update(**{field: value})
    with (
        connection.schema_editor() as editor,
        pytest.raises(RuntimeError, match="in_use"),
    ):
        migration.refuse_used_reverse(apps, editor)


def test_unused_projection_reverse_guard_allows_legacy_rows(legacy):
    migration = import_module(
        "apps.documents.migrations.0011_chunk_search_projection_v1"
    )
    with connection.schema_editor() as editor:
        migration.refuse_used_reverse(apps, editor)


def test_actual_schema_reverse_and_forward_preserve_unused_legacy_chunk(legacy):
    from django.db.migrations.executor import MigrationExecutor

    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    identifier = TextChunk.objects.get().pk
    try:
        MigrationExecutor(connection).migrate(
            [("apps_documents", "0010_embedding_generation_hnsw_v1")]
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT content FROM aquillm_textchunk WHERE id=%s", [identifier]
            )
            assert cursor.fetchone() == ("source",)
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='aquillm_textchunk' AND column_name='search_vector'"
            )
            assert cursor.fetchone() is None
    finally:
        MigrationExecutor(connection).migrate(leaves)
    assert TextChunk.objects.get(pk=identifier).search_vector_binding is None


def test_actual_used_schema_reverse_is_blocked_without_erasing_binding(legacy):
    from django.db.migrations.executor import MigrationExecutor

    publish(legacy)
    binding = TextChunk.objects.get().search_vector_binding
    leaves = MigrationExecutor(connection).loader.graph.leaf_nodes()
    try:
        with pytest.raises(RuntimeError, match="in_use"):
            MigrationExecutor(connection).migrate(
                [("apps_documents", "0010_embedding_generation_hnsw_v1")]
            )
        assert TextChunk.objects.get().search_vector_binding == binding
    finally:
        MigrationExecutor(connection).migrate(leaves)


def test_wrong_named_index_is_not_readiness_and_is_not_silently_replaced(legacy):
    migration = import_module(
        "apps.documents.migrations.0012_chunk_search_projection_gin_v1"
    )
    try:
        with connection.schema_editor(atomic=False) as editor:
            migration.drop_index(apps, editor)
            editor.execute(
                "CREATE INDEX textchunk_search_vector_gin_v1 "
                "ON aquillm_textchunk (doc_id)"
            )
        assert not index_ready("default")
        with (
            connection.schema_editor(atomic=False) as editor,
            pytest.raises(RuntimeError, match="unexpected"),
        ):
            migration.create_index(apps, editor)
    finally:
        with connection.schema_editor(atomic=False) as editor:
            migration.drop_index(apps, editor)
            migration.create_index(apps, editor)
