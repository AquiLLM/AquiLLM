# Persisted certificate fixtures and SQL observations for serving tests.

from apps.knowledge_graph.models import CollectionGraphMembershipCertificate
from apps.knowledge_graph.projection.membership_certificate_contracts import (
    native_membership_identity,
    sign_membership_certificate,
)
from apps.knowledge_graph.projection.membership_source_fingerprint import (
    load_membership_source_fingerprints,
)
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime


def certified_fixture(monkeypatch):
    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    settings.membership_certificate_enabled = True
    f.settings.membership_certificate_enabled = True
    publish_certificates(f)
    return f, runtime, settings, driver


def publish_certificates(f):
    fingerprints = load_membership_source_fingerprints(
        collection_ids=tuple(sorted(p.collection_id for p in f.projections.values())),
        using="default",
        check=lambda: None,
        require_fresh=True,
    )
    for projection in f.projections.values():
        projection.refresh_from_db()
        fp = fingerprints[projection.collection_id]
        payload = dict(
            format="membership-certificate-v1",
            projection_id=str(projection.pk),
            generation_key=str(projection.generation_key),
            collection_id=projection.collection_id,
            artifact_id=projection.artifact_id,
            schema_version=projection.schema_version,
            projection_version=projection.projection_version,
            identifier_key_version=projection.identifier_key_version,
            membership_epoch=projection.membership_epoch,
            membership_checksum=projection.membership_checksum,
            source_format="kg-membership-source-sql-v1",
            source_sha256=fp.input_sha256,
            entity_count=fp.entity_count,
            derivation_identity=native_membership_identity(f.codec),
        )
        CollectionGraphMembershipCertificate.objects.create(
            projection=projection,
            payload=payload,
            signature=sign_membership_certificate(payload, f.codec),
        )


def fingerprint_queries(queries):
    return [q["sql"] for q in queries if "kg-membership-source-sql-v1:" in q["sql"]]


def full_membership_queries(queries):
    return [
        q["sql"]
        for q in queries
        if "eligible" not in q["sql"]
        and "apps_knowledge_graph_collectionentity" in q["sql"]
        and 'AS "artifact__resolver_version"' in q["sql"]
    ]


def lifecycle_values(state):
    from datetime import timedelta

    from django.utils import timezone

    values = {"state": state}
    if state != "ready":
        values["ready_at"] = None
    if state == "building":
        values.update(
            lease_owner="test-worker",
            lease_expires_at=timezone.now() + timedelta(minutes=1),
        )
    return values
