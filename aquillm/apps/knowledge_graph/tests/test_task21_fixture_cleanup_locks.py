from __future__ import annotations

import pytest
from django.contrib.auth.models import User
from django.core.management import CommandError, call_command
from django.db import connection
from django.test import override_settings
from django.test.utils import CaptureQueriesContext

from apps.knowledge_graph.tests.task21_fixture_test_support import (
    FIXTURE_ID,
    HIDDEN_USERNAME,
    VISIBLE_USERNAME,
    cleanup,
    fixture_module,
    fixture_row_counts,
    manifest_checksum,
    seed,
    strict_eval_environment,
)

_STRICT_EVAL_ENVIRONMENT = strict_eval_environment


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_fails_closed_on_foreign_permission(tmp_path, monkeypatch) -> None:
    from apps.collections.models import CollectionPermission

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    foreign = User.objects.create_user(username="foreign-fixture-reference")
    CollectionPermission.objects.create(
        user=foreign,
        collection_id=payload["authorized_scope"][0]["collection_id"],
        permission="VIEW",
    )
    before = fixture_row_counts()
    with pytest.raises(CommandError, match="topology"):
        cleanup(manifest_path, payload)
    assert fixture_row_counts() == before
    assert manifest_path.exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_locks_canonical_order_and_uses_one_collection_delete_origin(
    tmp_path, monkeypatch
) -> None:
    from apps.collections.models import Collection

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    expected_ids = tuple(
        sorted({row["collection_id"] for row in payload["collections"].values()})
    )
    fixture_seed = fixture_module()
    events = []
    deleted_origins = []
    queryset_type = type(Collection.objects.all())
    original_select_for_update = queryset_type.select_for_update
    original_delete = queryset_type.delete

    def record_select_for_update(queryset, *args, **kwargs):
        if queryset.model is Collection:
            events.append(("row", None))
        return original_select_for_update(queryset, *args, **kwargs)

    def record_delete(queryset, *args, **kwargs):
        if queryset.model is Collection:
            deleted_origins.append(
                tuple(queryset.order_by("pk").values_list("pk", flat=True))
            )
        return original_delete(queryset, *args, **kwargs)

    monkeypatch.setattr(
        fixture_seed,
        "lock_collection_graph_advisory_scope",
        lambda collection_id: events.append(("advisory", collection_id)),
    )
    monkeypatch.setattr(queryset_type, "select_for_update", record_select_for_update)
    monkeypatch.setattr(queryset_type, "delete", record_delete)
    cleanup(manifest_path, payload)
    assert events[: len(expected_ids)] == [
        ("advisory", collection_id) for collection_id in expected_ids
    ]
    assert events[len(expected_ids)] == ("row", None)
    assert deleted_origins == [expected_ids]


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_bounds_owned_topology_discovery_queries(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    with CaptureQueriesContext(connection) as captured:
        cleanup(manifest_path, payload)
    selects = [
        query["sql"].upper()
        for query in captured.captured_queries
        if query["sql"].lstrip().upper().startswith("SELECT")
    ]

    def matching(table: str, limit: int):
        return [
            sql
            for sql in selects
            if f'FROM "{table.upper()}"' in sql and f"LIMIT {limit}" in sql
        ]

    assert matching("aquillm_collection", 6)
    assert matching("aquillm_collectionpermission", 6)
    assert len(matching("aquillm_rawtextdocument", 29)) >= 2
    assert matching("aquillm_textchunk", 31)
    assert matching("aquillm_documentfigure", 1)


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_requires_checksum_then_is_idempotent(tmp_path, monkeypatch) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    with pytest.raises(CommandError, match="checksum"):
        call_command(
            "seed_knowledge_graph_eval_fixture",
            "--cleanup",
            "--fixture-manifest",
            str(manifest_path),
            "--expected-manifest-checksum",
            "0" * 64,
        )
    assert fixture_row_counts() == (5, 28, 30, 5)
    first = cleanup(manifest_path, payload)
    assert "collections_deleted=5" in first
    assert not Collection.objects.filter(name__startswith=FIXTURE_ID).exists()
    assert not RawTextDocument.objects.filter(
        id__in=[row["document_id"] for row in payload["documents"].values()]
    ).exists()
    assert not TextChunk.objects.filter(
        pk__in=[row["chunk_id"] for row in payload["chunks"].values()]
    ).exists()
    assert not User.objects.filter(
        username__in=(VISIBLE_USERNAME, HIDDEN_USERNAME)
    ).exists()
    assert manifest_path.exists()
    second = cleanup(manifest_path, payload)
    assert "collections_deleted=0" in second
    assert manifest_checksum(payload) in second
    assert len(second) < 1_024
