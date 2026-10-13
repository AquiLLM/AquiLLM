"""Versioned build identities preserving exact absent-binding V1 hashes."""

import json
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class DocumentBuildIdentity:
    document_id: uuid.UUID
    source_hash: str
    ordered_chunk_signature: str
    extractor_package: str
    extractor_checkpoint: str
    extractor_model_revision: str
    extractor_config_checksum: str
    ontology_version: str
    ontology_checksum: str
    resolver_version: str
    resolver_checksum: str
    filter_version: str
    filter_checksum: str
    assembly_version: str
    assembly_checksum: str
    ontology_activation_signature: str = "0" * 64
    representation_digest: str | None = None

    def __post_init__(self) -> None:
        from .builds import _hash, _text

        if self.representation_digest is not None:
            _hash(self.representation_digest, "representation digest")
        if type(self.document_id) is not uuid.UUID or self.document_id.version is None:
            raise ValueError("document UUID must be an exact RFC 4122 UUID")
        for name in (
            "source_hash",
            "ordered_chunk_signature",
            "extractor_config_checksum",
            "ontology_checksum",
            "resolver_checksum",
            "filter_checksum",
            "assembly_checksum",
            "ontology_activation_signature",
        ):
            object.__setattr__(
                self,
                name,
                _hash(getattr(self, name), name.replace("_", " ")),
            )
        for name in (
            "extractor_package",
            "extractor_checkpoint",
            "extractor_model_revision",
            "ontology_version",
            "resolver_version",
            "filter_version",
            "assembly_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name.replace("_", " ")),
            )


@dataclass(frozen=True, slots=True)
class CollectionBuildIdentity:
    collection_id: int
    aggregate_source_signature: str
    extractor_version: str
    ontology_version: str
    ontology_checksum: str
    resolver_version: str
    resolver_checksum: str
    filter_version: str
    filter_checksum: str
    assembly_version: str
    assembly_checksum: str
    embedding_model_signature: str
    ontology_activation_signature: str = "0" * 64
    representation_digest: str | None = None

    def __post_init__(self) -> None:
        from .builds import _hash, _text

        if self.representation_digest is not None:
            _hash(self.representation_digest, "representation digest")
        if type(self.collection_id) is not int or not 0 < self.collection_id < 2**63:
            raise ValueError("collection id must be a positive database integer")
        for name in (
            "aggregate_source_signature",
            "ontology_checksum",
            "resolver_checksum",
            "filter_checksum",
            "assembly_checksum",
            "ontology_activation_signature",
        ):
            object.__setattr__(
                self,
                name,
                _hash(getattr(self, name), name.replace("_", " ")),
            )
        for name in (
            "extractor_version",
            "ontology_version",
            "resolver_version",
            "filter_version",
            "assembly_version",
            "embedding_model_signature",
        ):
            maximum = 512 if name == "embedding_model_signature" else 128
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name.replace("_", " "), maximum=maximum),
            )


def _identity_key(namespace: str, identity: object) -> str:
    payload = {
        "namespace": namespace,
        "identity": (
            asdict(identity)
            if hasattr(type(identity), "__dataclass_fields__")
            else identity
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def derive_document_build_key(identity: DocumentBuildIdentity) -> str:
    if type(identity) is not DocumentBuildIdentity:
        raise ValueError("document build identity must be exact")
    identity.__post_init__()
    return representation_build_key("kg-document-build", identity)


def derive_collection_build_key(identity: CollectionBuildIdentity) -> str:
    if type(identity) is not CollectionBuildIdentity:
        raise ValueError("collection build identity must be exact")
    identity.__post_init__()
    return representation_build_key("kg-collection-build", identity)


def representation_build_key(namespace, identity):
    payload = asdict(identity)
    if payload.pop("representation_digest") is None:
        return _identity_key(namespace + "-v1", payload)
    payload["representation_digest"] = identity.representation_digest
    encoded = json.dumps(
        {"version": namespace + "-v2", "identity": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode()
    return sha256(encoded).hexdigest()
