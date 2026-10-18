"""Fresh source evidence subordinate to the existing whole authority checks."""

from django.db import connections

from apps.knowledge_graph.projection import (
    membership_certificate_contracts as contracts,
)
from apps.knowledge_graph.projection import membership_source_fingerprint as source

# Eagerly capture the reader before a serving boundary can bless a replacement.
_SOURCE_BINDING = source._NATIVE_MEMBERSHIP_SOURCE_BINDING
_SOURCE_READER = _SOURCE_BINDING[0]


def _native_source():
    return (
        source.load_membership_source_fingerprints is _SOURCE_BINDING[0]
        and source._SQL is _SOURCE_BINDING[1]
        and source.MembershipSourceFingerprintV1 is _SOURCE_BINDING[2]
    )


def _certificate_rows(*, projection_ids, using, check):
    from apps.knowledge_graph.models import CollectionGraphMembershipCertificate

    from .ready_authority_reads_v2 import checked_rows

    return tuple(
        checked_rows(
            CollectionGraphMembershipCertificate.objects.using(using)
            .filter(projection_id__in=projection_ids)
            .only("projection_id", "payload", "signature")[: len(projection_ids) + 1],
            check,
        )
    )


def certified_membership_checksums(*, projections: tuple, using: str, codec, check):
    """Return positive H/C proofs; unsupported or missing evidence uses full F.

    Once a fresh source read starts, malformed results and original-context or
    deadline failures propagate. They are never converted into certificate misses.
    """
    check()
    identity = contracts.native_membership_identity(codec)
    if (
        not projections
        or identity is None
        or not _native_source()
        or connections[using].vendor != "postgresql"
    ):
        check()
        return {}
    if type(projections) is not tuple or len(projections) > 128:
        raise ValueError("membership certificate authority is unbounded")

    def proof_check():
        check()
        if (
            contracts.native_membership_identity(codec) != identity
            or not _native_source()
        ):
            raise ValueError("membership certificate native identity changed")

    proof_check()
    certificates = _certificate_rows(
        projection_ids=tuple(row.id for row in projections),
        using=using,
        check=proof_check,
    )
    by_projection = {}
    for row in certificates:
        proof_check()
        if row.projection_id in by_projection:
            raise ValueError("membership certificate projection duplicated")
        by_projection[row.projection_id] = row
    candidates = []
    for projection in projections:
        proof_check()
        certificate = by_projection.get(projection.id)
        if certificate is not None and contracts.verify_membership_certificate_binding(
            payload=certificate.payload,
            signature=certificate.signature,
            codec=codec,
            projection=projection,
        ):
            candidates.append(projection)
        proof_check()
    if not candidates:
        return {}
    collection_ids = tuple(sorted({row.collection_id for row in candidates}))
    fingerprints = _SOURCE_READER(
        collection_ids=collection_ids,
        using=using,
        check=proof_check,
        require_fresh=True,
    )
    proof_check()
    if set(fingerprints) != set(collection_ids):
        raise ValueError("membership fingerprint collection coverage invalid")
    if any(row.duplicate_links is not False for row in fingerprints.values()):
        raise ValueError("membership fingerprint duplicate links")
    result = {}
    for projection in candidates:
        proof_check()
        certificate = by_projection.get(projection.id)
        if certificate is not None and contracts.verify_membership_certificate(
            payload=certificate.payload,
            signature=certificate.signature,
            codec=codec,
            projection=projection,
            fingerprint=fingerprints[projection.collection_id],
        ):
            result[projection.id] = projection.membership_checksum
        proof_check()
    proof_check()
    return result
