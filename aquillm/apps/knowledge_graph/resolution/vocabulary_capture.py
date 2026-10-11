"""Bounded persisted capsules: exact pins and contents, never recaptured authority."""
from dataclasses import dataclass
import json
import os
from django.core.exceptions import ObjectDoesNotExist
from django.db import DatabaseError

from apps.collections.services.vocabulary_contracts import MAX_BYTES, digest, normalized
from apps.collections.services.vocabulary_database import vocabulary_database
from .vocabulary_policy import ScopedVetoV1, bind_digest

KEY = "vocabulary_veto_v1"


class VocabularyGraphUnavailable(ValueError):
    pass


def enabled():
    return os.environ.get("KG_VOCABULARY_VETO_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class VocabularyCapsuleV1:
    collection_id: int
    encoded: str
    policy: ScopedVetoV1


@dataclass(frozen=True, slots=True)
class VocabularyCaptureV1:
    selected: tuple
    capsules: tuple

    @property
    def policy(self):
        if not self.capsules:
            return None
        scopes = tuple(scope for c in self.capsules for scope in c.policy.scopes)
        return ScopedVetoV1(scopes, digest(tuple(c.policy.checksum for c in self.capsules)))

    def get(self, cid):
        if cid not in self.selected:
            raise VocabularyGraphUnavailable("vocabulary scope mismatch")
        return next((c for c in self.capsules if c.collection_id == cid), None)


def read_pins(selected, using, actor=None):
    """Keep graph imports independent of Django model initialization."""
    from apps.collections.services.vocabulary_access import read_pins as read
    return read(selected, using, actor)


def capture_build_vocabulary(collection_ids, *, using="default", actor=None):
    if not enabled():
        return None
    from apps.collections.services.vocabulary import published_entries
    selected = tuple(sorted(set(collection_ids)))
    if not selected:
        return VocabularyCaptureV1((), ())
    try:
        with vocabulary_database(using):
            pins, paths = read_pins(selected, using, actor)
            entries, _ = published_entries(pins, using)
            capsules = []
            total_bytes = 0
            for cid, path in paths:
                local = tuple(p for p in pins if p[2] == cid or (p[6] and p[2] in path[1:]))
                keys = {p[5] for p in local}
                vetoes = tuple(sorted({tuple(sorted((normalized(r["term"]),
                    normalized(r["expansions"][0])))) for key, r in entries
                    if key in keys and r["kind"] == "never_merge"}))
                if not vetoes:
                    continue
                payload = dict(version="scoped-vocabulary-v1", collection_id=cid,
                    pins=local, path=path, versions=tuple(sorted(keys)), never_merge=vetoes)
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                total_bytes += len(encoded.encode())
                if total_bytes > MAX_BYTES:
                    raise ValueError("aggregate vocabulary capsule limit")
                capsules.append(VocabularyCapsuleV1(cid, encoded,
                    ScopedVetoV1(((cid, vetoes),), digest(payload))))
            return VocabularyCaptureV1(selected, tuple(capsules))
    except (ValueError, PermissionError, ObjectDoesNotExist, DatabaseError) as exc:
        raise VocabularyGraphUnavailable("vocabulary policy unavailable") from exc


def capture_retrieval_vocabulary(authorization):
    if not enabled():
        return None
    from apps.collections.services.vocabulary import _authorization
    from apps.documents.services.source_loading import current_source_runtime
    runtime = current_source_runtime()
    if runtime is None or runtime.authorization is not authorization:
        raise VocabularyGraphUnavailable("original vocabulary runtime unavailable")
    with runtime.lock:
        try:
            selected = tuple(sorted(authorization.selected_collection_ids))
            with vocabulary_database(authorization.database_alias):
                actor = _authorization(authorization, selected)
                current = capture_build_vocabulary(selected, using=authorization.database_alias, actor=actor)
            captured = getattr(runtime, "_graph_vocabulary", None)
            if captured is not None and captured != current:
                raise VocabularyGraphUnavailable("captured vocabulary changed")
            if captured is None:
                runtime._graph_vocabulary = current
            runtime.observation["scoped_vocabulary"] = {
                "state": "observed", "digest": digest(tuple(c.policy.checksum for c in current.capsules))}
            return current
        except (ValueError, PermissionError, ObjectDoesNotExist, DatabaseError) as exc:
            runtime.observation["scoped_vocabulary"] = {"state": "unknown"}
            raise VocabularyGraphUnavailable("vocabulary policy unavailable") from exc


def metadata(capsule, base_checksum):
    if capsule is None:
        return {}
    return {KEY: dict(capsule=json.loads(capsule.encoded), base_checksum=base_checksum,
                     checksum=bind_digest(base_checksum, capsule.policy))}


def validate_artifact(artifact, capture, *, collection_id):
    value = artifact.get if type(artifact) is dict else lambda key, default=None: getattr(artifact, key, default)
    artifact_metadata = value("metadata")
    if artifact_metadata is not None and type(artifact_metadata) is not dict:
        raise VocabularyGraphUnavailable("vocabulary artifact metadata unavailable")
    stored = (artifact_metadata or {}).get(KEY)
    capsule = capture.get(collection_id) if capture is not None else None
    if capsule is None:
        if stored is not None:
            raise VocabularyGraphUnavailable("vocabulary artifact requires rebuild")
        return None
    if type(stored) is not dict or set(stored) != {"capsule", "base_checksum", "checksum"}:
        raise VocabularyGraphUnavailable("vocabulary artifact requires rebuild")
    expected = metadata(capsule, stored["base_checksum"])[KEY]
    if stored != expected or value("resolution_config_checksum") != expected["checksum"]:
        raise VocabularyGraphUnavailable("vocabulary artifact binding changed")
    if value("scope_type") == "document":
        from apps.knowledge_graph.models import graph_identity_checksum
        if stored["base_checksum"] != graph_identity_checksum("document-resolver", value("resolver_version")):
            raise VocabularyGraphUnavailable("vocabulary resolver identity changed")
    return capsule.policy


def current_artifact_policy(artifact, *, collection_id, using="default"):
    capture = capture_build_vocabulary((collection_id,), using=using)
    return validate_artifact(artifact, capture, collection_id=collection_id)


def effective_observation():
    from apps.documents.services.source_loading import current_source_runtime
    runtime = current_source_runtime()
    return dict(runtime.observation.get("scoped_vocabulary", {"state": "unknown"})) if runtime else {"state": "unknown"}
