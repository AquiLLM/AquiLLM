"""Authenticated source evidence for the exact supported membership derivation.

The keyholding application and default database owner remain trusted.
"""

import json
import re
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new

from . import identifiers as ids
from . import membership_checksum as checksums
from . import membership_identifiers as encoders
from . import (
    membership_source_query,
    memberships,
    records,
    serialization,
)
from .membership_derivation_source import (
    NATIVE_SOURCE_REVISION as _NATIVE_SOURCE_REVISION,
)

CERTIFICATE_FORMAT = "membership-certificate-v1"
SOURCE_FORMAT = "kg-membership-source-sql-v1"
_IDENTITY_DOMAIN = b"aquillm/membership-certificate/native-identity/v1\0"
_SIGNATURE_DOMAIN = b"aquillm/membership-certificate/signature/v1\0"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    (
        "format projection_id generation_key collection_id artifact_id schema_version "
        "projection_version identifier_key_version membership_epoch "
        "membership_checksum source_format source_sha256 entity_count "
        "derivation_identity"
    ).split()
)
_PROJECTION_FIELDS = tuple(
    (
        "generation_key collection_id artifact_id schema_version projection_version "
        "identifier_key_version membership_epoch membership_checksum"
    ).split()
)


def _binding():
    row = records.AutomaticCanonicalMembershipV1
    return (
        encoders._current_binding(),
        memberships.MEMBERSHIP_REGISTRY_GENERATION,
        memberships.load_automatic_membership_assignments,
        memberships.null_membership_decision_checksum,
        memberships.membership_decision_checksum,
        memberships._opaque_entity_key,
        memberships._opaque_membership_key,
        memberships._checked_rows,
        memberships._load_membership_source_rows,
        memberships._membership_check,
        memberships._batch_size,
        memberships._collection_ids,
        memberships.AutomaticCanonicalMembershipV1,
        memberships.sha256,
        memberships.json.dumps,
        memberships.ProjectionIdentifierDomain,
        encoders.MembershipKeyEncoder,
        encoders.MembershipKeyEncoder.__init__,
        encoders.MembershipKeyEncoder._integer,
        encoders.MembershipKeyEncoder._discard,
        encoders.MembershipKeyEncoder._ready,
        encoders._generation_state,
        encoders.Domain,
        encoders.MembershipKeyEncoder.entity,
        encoders.MembershipKeyEncoder.canonical,
        encoders.MembershipKeyEncoder._encoded,
        checksums.membership_decision_checksum,
        checksums._ordered_rows,
        checksums._payload,
        checksums._checkpoint,
        checksums._FIELDS,
        checksums.AutomaticCanonicalMembershipV1,
        checksums._SCALARS,
        checksums._encode,
        checksums._provider_types,
        checksums._BATCH_SIZE,
        checksums.json.dumps,
        checksums.sha256,
        serialization._encode,
        serialization.projection_checksum,
        serialization._provider_types,
        row,
        row.__init__,
        row.__post_init__,
        row.__getattribute__,
        row.__new__,
        records._key,
        records._token,
        records._count,
        records.fields,
        records.NULLABLE_RECORD_FIELDS,
        serialization._key,
        serialization._token,
        serialization._count,
        serialization._uuid,
        tuple(getattr(row, name) for name in row.__dataclass_fields__),
    )


# Import eagerly: a later first certificate check must not bless replacements.
_NATIVE_BINDING = _binding()
_NATIVE_IDS = ids._NATIVE_HMAC_ENCODING_BINDING
_DEFINING_BINDINGS = tuple(
    (module, dict(module._NATIVE_MEMBERSHIP_BINDING))
    for module in (
        memberships,
        checksums,
        encoders,
        serialization,
        records,
        membership_source_query,
    )
)


def _same_binding(current, native):
    if type(native) is tuple:
        return (
            type(current) is tuple
            and len(current) == len(native)
            and all(_same_binding(a, b) for a, b in zip(current, native))
        )
    if type(native) in (str, int, frozenset):
        return type(current) is type(native) and current == native
    return current is native


def _capture(codec):
    try:
        if _NATIVE_SOURCE_REVISION is None:
            return None
        for module, bindings in _DEFINING_BINDINGS:
            for path, native in bindings.items():
                current = module
                for part in path.split("."):
                    current = getattr(current, part)
                if not _same_binding(current, native):
                    return None
        current = _binding()
        if (
            type(codec) is not _NATIVE_IDS[0]
            or not _same_binding(current, _NATIVE_BINDING)
            or any(a is not b for a, b in zip(current[0], _NATIVE_IDS))
        ):
            return None
        fields = codec._key, codec.key_version, codec.codec_version
        if (
            type(fields[0]) is not bytes
            or not fields[0]
            or fields[2] != "projection-id-v1"
        ):
            return None
        serialization._token(fields[1], "key_version", 64)
        return fields
    except (AttributeError, TypeError, ValueError):
        return None


def _canonical(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(fields):
    body = [
        "membership-derivation-record-checksum-v1",
        "canonical-json-utf8-v1",
        _NATIVE_SOURCE_REVISION,
        _DEFINING_BINDINGS[0][1]["MEMBERSHIP_REGISTRY_GENERATION"],
        fields[1],
        fields[2],
    ]
    return hmac_new(fields[0], _IDENTITY_DOMAIN + _canonical(body), sha256).hexdigest()


def native_membership_identity(codec) -> str | None:
    fields = _capture(codec)
    if fields is None:
        return None
    identity = _identity(fields)
    return identity if _capture(codec) == fields else None


def validate_membership_certificate_payload(payload):
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise ValueError("membership certificate fields are invalid")
    if (
        payload["format"] != CERTIFICATE_FORMAT
        or payload["source_format"] != SOURCE_FORMAT
    ):
        raise ValueError("membership certificate format is unsupported")
    for name in ("projection_id", "generation_key"):
        serialization._uuid(payload[name], name)
    for name in ("collection_id", "artifact_id", "membership_epoch", "entity_count"):
        serialization._count(
            payload[name],
            name,
            minimum=1 if name.endswith("_id") else 0,
            maximum=2**63 - 1,
        )
    for name in ("schema_version", "projection_version", "identifier_key_version"):
        serialization._token(payload[name], name, 64)
    for name in ("membership_checksum", "source_sha256", "derivation_identity"):
        if type(payload[name]) is not str or _HEX.fullmatch(payload[name]) is None:
            raise ValueError("membership certificate digest is invalid")


def sign_membership_certificate(payload: dict, codec) -> str:
    fields = _capture(codec)
    if type(payload) is dict:
        payload = payload.copy()
    validate_membership_certificate_payload(payload)
    if (
        fields is None
        or payload["derivation_identity"] != _identity(fields)
        or payload["identifier_key_version"] != fields[1]
    ):
        raise ValueError("membership certificate native identity is unavailable")
    signature = hmac_new(
        fields[0], _SIGNATURE_DOMAIN + _canonical(payload), sha256
    ).hexdigest()
    if _capture(codec) != fields:
        raise ValueError("membership certificate native identity changed")
    return signature


def verify_membership_certificate_binding(
    *, payload: dict, signature: str, codec, projection
) -> bool:
    """Authenticate an occurrence before spending a fresh source fingerprint."""
    fields = _capture(codec)
    if fields is None:
        return False
    try:
        if type(payload) is dict:
            payload = payload.copy()
        validate_membership_certificate_payload(payload)
        if type(signature) is not str or _HEX.fullmatch(signature) is None:
            return False
        expected = sign_membership_certificate(payload, codec)
        if not compare_digest(signature, expected):
            return False
        if (
            projection.state != "ready"
            or str(projection.id) != payload["projection_id"]
        ):
            return False
        for name in _PROJECTION_FIELDS:
            value = getattr(projection, name)
            if name == "generation_key":
                value = str(value)
            if type(value) is not type(payload[name]) or value != payload[name]:
                return False
        return _capture(codec) == fields
    except (AttributeError, TypeError, ValueError):
        return False


def verify_membership_certificate(
    *, payload: dict, signature: str, codec, projection, fingerprint
) -> bool:
    fields = _capture(codec)
    if fields is None:
        return False
    try:
        if type(payload) is dict:
            payload = payload.copy()
        if not verify_membership_certificate_binding(
            payload=payload, signature=signature, codec=codec, projection=projection
        ):
            return False
        if (
            fingerprint.duplicate_links is not False
            or type(fingerprint.collection_id) is not int
            or type(fingerprint.entity_count) is not int
            or type(fingerprint.source_rows) is not int
            or fingerprint.collection_id != payload["collection_id"]
            or fingerprint.input_sha256 != payload["source_sha256"]
            or fingerprint.entity_count != payload["entity_count"]
            or fingerprint.source_rows != fingerprint.entity_count
        ):
            return False
        return _capture(codec) == fields
    except (AttributeError, TypeError, ValueError):
        return False
