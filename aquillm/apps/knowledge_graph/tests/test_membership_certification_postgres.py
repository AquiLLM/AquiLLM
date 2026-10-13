"""Two real connections demonstrate torn RC inputs and bound RR inputs."""
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from django.db import close_old_connections, connection, connections, transaction
from django.utils import timezone

from apps.knowledge_graph.models import CollectionGraphMembershipCertificate, CollectionGraphProjection
from apps.knowledge_graph.projection import membership_certification as cert
from apps.knowledge_graph.projection.database_deadline import projection_operation
from apps.knowledge_graph.projection.membership_certificate_contracts import verify_membership_certificate
from apps.knowledge_graph.projection.membership_source_fingerprint import load_membership_source_fingerprints
from apps.knowledge_graph.tests.test_projection_canonical_resolver_postgres import CODEC, source_rows

pytestmark = [pytest.mark.django_db(transaction=True), pytest.mark.skipif(
    os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1", reason="isolated PostgreSQL only")]


def mutate(link):
    def run():
        close_old_connections()
        try:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE apps_knowledge_graph_canonicalentitylink SET decision_checksum=%s WHERE id=%s",
                               ["e" * 64, link.pk])
        finally:
            close_old_connections()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(run).result(timeout=10)


def full(cid):
    return cert.membership_decision_checksum(cert.load_automatic_membership_assignments(
        collection_ids=(cid,), using="default", batch_size=100, codec=CODEC, check=lambda: None), check=lambda: None)


def test_read_committed_can_be_torn_between_H_and_F(source_rows):
    entity, _, link, _, expected = source_rows
    with transaction.atomic():
        before = load_membership_source_fingerprints(collection_ids=(entity.collection_id,), using="default", check=lambda: None)
        mutate(link)
        assert full(entity.collection_id) != expected
    after = load_membership_source_fingerprints(collection_ids=(entity.collection_id,), using="default", check=lambda: None)
    assert before != after


def test_top_level_read_only_RR_binds_H_and_full_F(source_rows):
    entity, _, link, _, expected = source_rows
    with projection_operation(10000, call_timeout_ms=5000) as operation:
        with cert.membership_source_snapshot(using="default", operation=operation):
            with connection.cursor() as cursor:
                cursor.execute("SHOW transaction_read_only")
                assert cursor.fetchone() == ("on",)
            before = load_membership_source_fingerprints(collection_ids=(entity.collection_id,), using="default", check=operation.remaining_ms, require_fresh=False)
            mutate(link)
            assert full(entity.collection_id) == expected
            assert load_membership_source_fingerprints(collection_ids=(entity.collection_id,), using="default", check=operation.remaining_ms, require_fresh=False) == before
    assert full(entity.collection_id) != expected


@pytest.fixture
def ready(source_rows, monkeypatch):
    entity, _, link, projection, expected = source_rows
    CollectionGraphProjection.objects.filter(pk=projection.pk).update(
        state="ready", ready_at=timezone.now(), lease_owner="", lease_expires_at=None,
        graph_checksum="a" * 64, snapshot_checksum="a" * 64)
    # Fixture's legacy artifact has no FK scope, as production ready CAS forbids.
    with connection.cursor() as cursor:
        cursor.execute("UPDATE apps_knowledge_graph_graphartifact SET collection_scope_id=%s WHERE id=%s",
                       [entity.collection_id, entity.artifact_id])
    projection.refresh_from_db()
    monkeypatch.setattr(cert, "projection_identifier_codec", lambda settings: CODEC)
    monkeypatch.setattr(connections._connections, "certificate_state_test", connections["default"], raising=False)
    settings = SimpleNamespace(membership_certificate_enabled=True, projection_job_timeout_ms=20000,
                               projection_timeout_ms=5000, projection_batch_size=100)
    return entity, link, projection, settings


def certify(ready):
    _, _, projection, settings = ready
    return cert.certify_ready_membership(projection_id=projection.pk, settings=settings,
        source_using="default", state_using="certificate_state_test")


def test_certification_publishes_after_source_cleanup_and_fresh_H_rejects_mutation(ready, monkeypatch):
    original = cert.publish_membership_certificate
    def publish(**kwargs):
        assert not connection.in_atomic_block or kwargs["using"] == "certificate_state_test"
        # State database wrapper is allowed; the old source RR is already gone.
        with connection.cursor() as cursor:
            cursor.execute("SHOW transaction_isolation")
            assert cursor.fetchone() == ("read committed",)
        return original(**kwargs)
    monkeypatch.setattr(cert, "publish_membership_certificate", publish)
    assert certify(ready)
    entity, link, projection, _ = ready
    certificate = CollectionGraphMembershipCertificate.objects.get(projection=projection)
    fresh = lambda: load_membership_source_fingerprints(collection_ids=(entity.collection_id,), using="default", check=lambda: None)[entity.collection_id]
    assert verify_membership_certificate(payload=certificate.payload, signature=certificate.signature,
        codec=CODEC, projection=projection, fingerprint=fresh())
    mutate(link)
    assert not verify_membership_certificate(payload=certificate.payload, signature=certificate.signature,
        codec=CODEC, projection=projection, fingerprint=fresh())


def test_full_checksum_mismatch_publishes_nothing(ready):
    mutate(ready[1])
    assert not certify(ready)
    assert not CollectionGraphMembershipCertificate.objects.exists()


def test_rebuilt_occurrence_fences_state_CAS(ready, monkeypatch):
    original = cert.publish_membership_certificate
    def publish(**kwargs):
        CollectionGraphProjection.objects.filter(pk=ready[2].pk).update(
            state="superseded", superseded_at=timezone.now())
        return original(**kwargs)
    monkeypatch.setattr(cert, "publish_membership_certificate", publish)
    assert not certify(ready)
    assert not CollectionGraphMembershipCertificate.objects.exists()


@pytest.mark.parametrize("failure", ["timeout", "revocation", "native"])
def test_cleanup_failure_or_revocation_never_publishes(ready, monkeypatch, failure):
    from contextlib import contextmanager
    original = cert.membership_source_snapshot
    @contextmanager
    def snapshot(**kwargs):
        with original(**kwargs):
            yield
        assert not connection.in_atomic_block
        if failure == "timeout":
            raise TimeoutError("expired")
        if failure == "revocation":
            kwargs["operation"].close()
        else:
            monkeypatch.setattr(cert, "native_membership_identity", lambda codec: None)
    monkeypatch.setattr(cert, "membership_source_snapshot", snapshot)
    with pytest.raises((TimeoutError, RuntimeError)):
        certify(ready)
    assert not CollectionGraphMembershipCertificate.objects.exists()


def test_mutation_between_snapshot_and_CAS_is_rejected_by_fresh_H(ready, monkeypatch):
    original = cert.publish_membership_certificate
    def publish(**kwargs):
        mutate(ready[1])
        return original(**kwargs)
    monkeypatch.setattr(cert, "publish_membership_certificate", publish)
    assert certify(ready)
    entity, _, projection, _ = ready
    certificate = CollectionGraphMembershipCertificate.objects.get(projection=projection)
    fingerprint = load_membership_source_fingerprints(collection_ids=(entity.collection_id,),
        using="default", check=lambda: None)[entity.collection_id]
    assert not verify_membership_certificate(payload=certificate.payload, signature=certificate.signature,
        codec=CODEC, projection=projection, fingerprint=fingerprint)
