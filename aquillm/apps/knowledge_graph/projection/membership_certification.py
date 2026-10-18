"""Bounded full membership proof, captured in one read-only source snapshot."""

from contextlib import contextmanager
from uuid import UUID

from django.db import connections, transaction

from apps.knowledge_graph.models import (
    CollectionGraphMembershipCertificate, CollectionGraphMembershipState,
    CollectionGraphProjection,
)
from .database_deadline import current_operation, projection_database, projection_operation
from .membership_certificate_contracts import native_membership_identity, sign_membership_certificate
from .membership_certificate_store import publish_membership_certificate
from .membership_source_fingerprint import SOURCE_FINGERPRINT_FORMAT, load_membership_source_fingerprints
from .memberships import load_automatic_membership_assignments, membership_decision_checksum
from .runtime import load_projection_runtime_settings, projection_identifier_codec


@contextmanager
def membership_source_snapshot(*, using, operation):
    connection = connections[using]
    if connection.in_atomic_block or not connection.get_autocommit():
        raise RuntimeError("membership_certificate_nested_source_transaction")
    native = getattr(connection, "connection", None)
    if native is not None:
        status = (native.get_transaction_status() if hasattr(native, "get_transaction_status")
                  else native.info.transaction_status)
        if status != 0:  # Both supported psycopg adapters use zero for IDLE.
            raise RuntimeError("membership_certificate_nested_source_transaction")
    operation.remaining_ms()
    with transaction.atomic(using=using):
        # SET TRANSACTION must precede projection_database's pg_settings SELECT.
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        operation.remaining_ms()
        with projection_database(using, operation=operation):
            yield
            operation.remaining_ms()
    operation.remaining_ms()


def certify_ready_membership(*, projection_id: UUID, settings=None,
                             source_using="projection_source", state_using="projection_state") -> bool:
    if type(projection_id) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    settings = load_projection_runtime_settings() if settings is None else settings
    if not (getattr(settings, "membership_certificate_enabled", False)
            or getattr(settings, "membership_certificate_issuance_enabled", False)):
        return False
    if source_using == state_using:
        raise ValueError("membership certificate aliases must be distinct")
    codec = projection_identifier_codec(settings)
    identity = native_membership_identity(codec)
    if identity is None:
        return False
    with projection_operation(min(settings.projection_job_timeout_ms, 300000),
                              call_timeout_ms=min(settings.projection_timeout_ms, 5000)) as operation:
        def check():
            if current_operation() is not operation:
                raise RuntimeError("membership_certificate_operation_replaced")
            operation.remaining_ms()
            if native_membership_identity(codec) != identity:
                raise RuntimeError("membership_certificate_native_changed")

        with membership_source_snapshot(using=source_using, operation=operation):
            check()
            projection = (CollectionGraphProjection.objects.using(source_using)
                .filter(pk=projection_id, state="ready", artifact__status="active",
                        artifact__scope_type="collection", artifact__evaluation_only=False)
                .select_related("artifact").first())
            check()
            if projection is None or projection.collection_id is None:
                return False
            cid = projection.collection_id
            if (projection.collection_pk_snapshot != cid
                    or projection.artifact_pk_snapshot != projection.artifact_id
                    or projection.artifact.collection_scope_id != cid):
                return False
            state = CollectionGraphMembershipState.objects.using(source_using).filter(collection_id=cid).first()
            check()
            if (state is None or state.active_artifact_id != projection.artifact_id
                    or state.registry_epoch != projection.membership_epoch
                    or state.membership_checksum != projection.membership_checksum):
                return False
            previous = (CollectionGraphMembershipCertificate.objects.using(source_using)
                        .filter(projection_id=projection_id).values_list("signature", flat=True).first())
            check()
            fingerprint = load_membership_source_fingerprints(
                collection_ids=(cid,), using=source_using, check=check, require_fresh=False)[cid]
            if fingerprint.duplicate_links:
                return False
            assignments = load_automatic_membership_assignments(
                collection_ids=(cid,), using=source_using, batch_size=settings.projection_batch_size,
                codec=codec, check=check)
            checksum = membership_decision_checksum(assignments, check=check)
            if (checksum != projection.membership_checksum or checksum != state.membership_checksum
                    or len(assignments) != fingerprint.entity_count):
                return False
            payload = dict(format="membership-certificate-v1", projection_id=str(projection.pk),
                generation_key=str(projection.generation_key), collection_id=cid,
                artifact_id=projection.artifact_id, schema_version=projection.schema_version,
                projection_version=projection.projection_version,
                identifier_key_version=projection.identifier_key_version,
                membership_epoch=projection.membership_epoch, membership_checksum=checksum,
                source_format=SOURCE_FINGERPRINT_FORMAT, source_sha256=fingerprint.input_sha256,
                entity_count=fingerprint.entity_count, derivation_identity=identity)
            signature = sign_membership_certificate(payload, codec)
            check()
        # The source transaction is closed before the restricted state CAS.
        check()
        with projection_database(state_using, operation=operation):
            check()
            return publish_membership_certificate(payload=payload, signature=signature,
                previous_signature=previous, using=state_using, check=check)
