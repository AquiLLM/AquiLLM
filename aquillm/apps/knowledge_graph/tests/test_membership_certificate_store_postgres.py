"""Restricted publication, exact authority CAS and migration roundtrip."""

import json
import os

import pytest
from django.db import DatabaseError, connection, transaction
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.models import CollectionGraphProjection
from apps.knowledge_graph.tests.test_projection_locking_postgres import (
    _active_artifact,
    _building_projection,
)

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="isolated PostgreSQL only",
    ),
]


@pytest.fixture
def certificate_payload(monkeypatch):
    monkeypatch.setenv("KG_PROJECTION_IDENTIFIER_KEY_VERSION", "certificate-test-v1")
    collection = Collection.objects.create(name="membership certificate")
    artifact = _active_artifact(collection)
    _, projection = _building_projection(collection, artifact, timezone.now())
    CollectionGraphProjection.objects.filter(pk=projection.pk).update(
        state="ready",
        ready_at=timezone.now(),
        lease_owner="",
        lease_expires_at=None,
        graph_checksum="a" * 64,
        snapshot_checksum="a" * 64,
    )
    projection.refresh_from_db()
    return dict(
        format="membership-certificate-v1",
        projection_id=str(projection.pk),
        generation_key=str(projection.generation_key),
        collection_id=collection.pk,
        artifact_id=artifact.pk,
        schema_version=projection.schema_version,
        projection_version=projection.projection_version,
        identifier_key_version=projection.identifier_key_version,
        membership_epoch=projection.membership_epoch,
        membership_checksum=projection.membership_checksum,
        source_format="kg-membership-source-sql-v1",
        source_sha256="b" * 64,
        entity_count=projection.entity_count,
        derivation_identity="c" * 64,
    )


def publish(
    payload, signature="d" * 64, previous=None, role="aquillm_projection_state"
):
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE " + role)
        cursor.execute(
            "SELECT public.kg_publish_membership_certificate_v1(%s::jsonb,%s,%s)",
            [json.dumps(payload), signature, previous],
        )
        return cursor.fetchone()[0]


def test_absence_prior_signature_and_identical_idempotence(certificate_payload):
    payload = certificate_payload
    assert publish(payload)
    assert publish(payload)
    replacement = dict(payload, source_sha256="e" * 64)
    assert not publish(replacement, "f" * 64)
    assert not publish(replacement, "f" * 64, "0" * 64)
    assert publish(replacement, "f" * 64, "d" * 64)
    assert not publish(payload, "d" * 64, "d" * 64)


@pytest.mark.parametrize(
    "change",
    [
        {"membership_epoch": 999},
        {"generation_key": "00000000-0000-0000-0000-000000000099"},
        {"membership_checksum": "e" * 64},
    ],
)
def test_stale_identity_is_rejected(certificate_payload, change):
    assert not publish(dict(certificate_payload, **change))


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_id": True},
        {"entity_count": 1.0},
        {"source_sha256": "bad"},
        {"extra": "raw"},
        {"projection_id": "bad"},
    ],
)
def test_forged_shape_is_rejected(certificate_payload, change):
    assert not publish(dict(certificate_payload, **change))


def test_source_cannot_publish_and_public_has_no_execute(certificate_payload):
    with pytest.raises(DatabaseError):
        publish(certificate_payload, role="aquillm_projection_source")
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_proc p, LATERAL "
            "aclexplode(COALESCE(p.proacl,acldefault('f',p.proowner))) a "
            "WHERE p.oid="
            "'public.kg_publish_membership_certificate_v1(jsonb,text,text)'"
            "::regprocedure "
            "AND a.grantee=0 AND a.privilege_type='EXECUTE')"
        )
        assert cursor.fetchone() == (False,)
    with transaction.atomic(), connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE aquillm_projection_source")
        cursor.execute(
            "SELECT payload FROM "
            "public.apps_knowledge_graph_collectiongraphmembershipcertificate"
        )
    with (
        pytest.raises(DatabaseError),
        transaction.atomic(),
        connection.cursor() as cursor,
    ):
        cursor.execute("SET LOCAL ROLE aquillm_projection_source")
        cursor.execute(
            "DELETE FROM "
            "public.apps_knowledge_graph_collectiongraphmembershipcertificate"
        )


def test_concurrent_prior_signature_allows_one_replacement(certificate_payload):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from django.db import close_old_connections

    assert publish(certificate_payload)
    gate = Barrier(2)

    def replace(digest):
        close_old_connections()
        try:
            gate.wait(timeout=10)
            return publish(
                dict(certificate_payload, source_sha256=digest * 64),
                digest * 64,
                "d" * 64,
            )
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(replace, value) for value in ("e", "f")]
        assert sorted(future.result(timeout=20) for future in futures) == [False, True]


def test_changed_state_and_missing_grant_reject_publication(certificate_payload):
    from apps.knowledge_graph.models import CollectionGraphMembershipState

    CollectionGraphMembershipState.objects.filter(
        collection_id=certificate_payload["collection_id"]
    ).update(registry_epoch=99)
    assert not publish(certificate_payload)
    with pytest.raises(DatabaseError), transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(
                "REVOKE EXECUTE ON FUNCTION "
                "public.kg_publish_membership_certificate_v1(jsonb,text,text) "
                "FROM aquillm_projection_state"
            )
        publish(certificate_payload)


def test_migration_backward_forward_and_state_consistency():
    from django.db.migrations.executor import MigrationExecutor

    before = [("apps_knowledge_graph", "0015_projection_chunk_input_lookup")]
    after = [("apps_knowledge_graph", "0016_membership_certificate")]
    executor = MigrationExecutor(connection)
    try:
        executor.migrate(before)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass("
                "'public.apps_knowledge_graph_collectiongraphmembershipcertificate'), "
                "to_regprocedure("
                "'public.kg_publish_membership_certificate_v1(jsonb,text,text)')"
            )
            assert cursor.fetchone() == (None, None)
    finally:
        MigrationExecutor(connection).migrate(after)
    executor = MigrationExecutor(connection)
    state = executor.loader.project_state(after)
    model = state.apps.get_model(
        "apps_knowledge_graph", "CollectionGraphMembershipCertificate"
    )
    assert model._meta.pk.name == "projection"
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT has_function_privilege('aquillm_projection_state', "
            "'public.kg_publish_membership_certificate_v1(jsonb,text,text)', 'EXECUTE')"
        )
        assert cursor.fetchone() == (True,)
