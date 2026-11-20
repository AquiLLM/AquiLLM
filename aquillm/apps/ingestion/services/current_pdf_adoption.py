"""Trusted present correspondence for one existing PDF owner, never its history.

Only this service captures attached bytes and executes the fixed validation
policy. It returns the existing immutable OutputResultV1 and never publishes a
document or representation. Immutable blobs/source rows written before a failed
final binding may remain unreferenced; they are not a successful adoption.

The finite shared deadline bounds SQL, storage and isolated parsing. An operator
process must provide the outer wall watchdog; this module installs no signals.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import math
import posixpath
import time

from django.contrib.auth.models import User
from django.db import DatabaseError, connections
from django.db.models import F, Func, IntegerField, Q
from django.db.models.functions import Length
from storages.backends.s3 import S3Storage

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import (
    DESCENDED_FROM_DOCUMENT, DocumentRepresentationGeneration,
    DocumentRepresentationState, PDFDocument,
)
from apps.documents.services.representation_contracts import DocumentRefV1, SourceScopeUnavailable
from apps.documents.services.representation_store import SourceSHA256, document_payload
from apps.knowledge_graph.graph.invalidation import _lock_document_scope, locked_collection_parent_rows
from apps.ingestion.models import ExtractionRevision, IngestionBatch, IngestionBatchItem, IngestionOutput
from lib.parsers import pdf_current_validation
from lib.parsers.source_contracts import ExtractionV1, SourceElementV1
from . import extraction_revisions, source_artifacts, source_storage
from .current_adoption_database import bounded_database as _bounded_database, remaining_ms as _remaining
from .output_ledger import OutputResultV1
from .output_payloads import MAX_TEXT, output_key
from .output_status import item_snapshot


def _unavailable():
    raise SourceScopeUnavailable("source_scope_unavailable")


def _parents(collection_id, *, using):
    chain = []
    while collection_id is not None and len(chain) < 100:
        row = Collection.objects.using(using).filter(pk=collection_id).values_list("pk", "parent_id").first()
        if row is None or row[0] in {key for key, _ in chain}:
            _unavailable()
        chain.append(row)
        collection_id = row[1]
    if collection_id is not None:
        _unavailable()
    return tuple(chain)


def _authorize(actor_id, chain, *, using, locked=False):
    actors = User.objects.using(using).filter(pk=actor_id, is_active=True)
    grants = CollectionPermission.objects.using(using).filter(user_id=actor_id,
        collection_id__in=[key for key, _ in chain], permission__in=("EDIT", "MANAGE"))
    if locked:
        # NO KEY UPDATE serializes active-state changes while permitting normal
        # FK checks. Grant rows remain locked through the binding commit.
        actors = actors.select_for_update(no_key=True)
        grants = grants.select_for_update().order_by("pk")
    if actors.only("pk", "is_active").first() is None or not list(grants.values_list("pk", flat=True)):
        _unavailable()


def _unique_legacy_owner(ref, *, using):
    owners = []
    for model in DESCENDED_FROM_DOCUMENT:
        owners.extend((model._meta.label_lower, pk) for pk in
            model._base_manager.using(using).filter(id=ref.id).values_list("pk", flat=True)[:2])
    if owners != [(ref.model_label, ref.pkid)]:
        _unavailable()
    states = list(DocumentRepresentationState.objects.using(using).filter(
        Q(model_label=ref.model_label, document_pkid=ref.pkid) | Q(document_id=ref.id))
        .values("pk", "model_label", "document_pkid", "document_id", "collection_id", "active_generation_id")[:2])
    if len(states) > 1:
        _unavailable()
    if states:
        state = states[0]
        if (state["model_label"], state["document_pkid"], state["document_id"], state["collection_id"]) != (
                ref.model_label, ref.pkid, ref.id, ref.collection_id) or state["active_generation_id"]:
            _unavailable()
        if DocumentRepresentationGeneration.objects.using(using).filter(
                owner_id=state["pk"], status__in=("building", "staged", "active")).exists():
            _unavailable()


def _owner(ref, *, using, locked=False):
    query = PDFDocument.objects.using(using).filter(pkid=ref.pkid, id=ref.id,
        collection_id=ref.collection_id, full_text_hash=ref.source_hash).annotate(
            _points=Length("full_text"),
            _bytes=Func(F("full_text"), template="octet_length(convert_to(%(expressions)s, 'UTF8'))",
                        output_field=IntegerField()),
            _canonical=SourceSHA256(F("full_text")))
    # The first query transfers metadata only, including for malformed legacy
    # rows. Repeat all predicates in the body query to close growth/edit races.
    sizes = query.values("_points", "_bytes", "_canonical").first()
    if (sizes is None or not 0 < sizes["_points"] <= MAX_TEXT
            or not 0 < sizes["_bytes"] <= 8 * 1024 * 1024 or sizes["_canonical"] != ref.source_hash):
        _unavailable()
    query = query.filter(_points__gt=0, _points__lte=MAX_TEXT,
        _bytes__gt=0, _bytes__lte=8 * 1024 * 1024, _canonical=ref.source_hash)
    if locked:
        query = query.select_for_update()
    owner = query.first()
    if (owner is None or not owner.full_text.strip() or "\0" in owner.full_text
            or sha256(owner.full_text.encode("utf-8")).hexdigest() != ref.source_hash):
        _unavailable()
    return owner


def _attachment(owner):
    name = owner.pdf_file.name
    source_storage._name(name)
    storage = source_storage._backend(owner.pdf_file.storage)
    if type(storage) is S3Storage:
        identity = {field: getattr(storage, field) for field in (
            "bucket_name", "location", "endpoint_url", "region_name", "use_ssl", "verify",
            "access_key", "secret_key", "security_token", "session_profile")}
        identity["object_parameters"] = storage.get_object_parameters(storage._normalize_name(name))
        identity["client_config"] = vars(storage.client_config) if storage.client_config is not None else None
    else:
        identity = {"location": storage.location}
    # This private in-memory identity is never returned, logged, or persisted as
    # evidence. Object credentials/configuration must not change during capture.
    return storage, name, deepcopy(identity)


def _current(ref, actor_id, *, using, locked=False):
    chain = _parents(ref.collection_id, using=using)
    _authorize(actor_id, chain, using=using, locked=locked)
    _unique_legacy_owner(ref, using=using)
    owner = _owner(ref, using=using, locked=locked)
    storage, name, backend = _attachment(owner)
    return owner, chain, (storage, name, backend)


def _same_current(first, current):
    before, chain, attached = first
    after, fresh_chain, fresh_attached = current
    if (chain != fresh_chain or attached != fresh_attached
            or before.full_text != after.full_text or before.title != after.title):
        _unavailable()


def _payload(owner, chain, artifact):
    return {"title": owner.title, "full_text": owner.full_text, "normalized_type": "pdf", "modality": "text",
        "provider": "", "model": "", "issue_codes": ["structure_unavailable"], "quality": "partial",
        "source_artifact": asdict(artifact), "media_artifact": None, "source_content_type": "application/pdf",
        "source_format": "pdf", "figure_index": 0, "extracted_caption": "", "location_metadata": {},
        "collection_id": owner.collection_id, "collection_parent_id": chain[0][1]}


def _origins(ref, *, using):
    return list(IngestionOutput.objects.using(using).filter(status="success", document_model=ref.model_label,
        document_pkid=ref.pkid, document_id=ref.id).select_related("item__batch")[:2])


def _identical(row, ref, payload, evidence, occurrence, revision):
    # A retry performs its own current policy/source/permission checks. Original
    # actor/time describe the first operation and remain immutable; they are not
    # credentials and do not need to equal this freshly authorized operator.
    retained = row.provenance_evidence
    comparable = {key: value for key, value in retained.items() if key not in ("actor_id", "validated_at")}
    wanted = {key: value for key, value in evidence.items() if key not in ("actor_id", "validated_at")}
    if (row.provenance_kind != "current_adoption" or comparable != wanted or row.payload != payload
            or row.source_occurrence_id != occurrence.occurrence_id or row.extraction_revision_id != revision.pk
            or (row.document_collection_id, row.document_source_hash) != (ref.collection_id, ref.source_hash)
            or row.output_kind != "text" or row.element_key != "body" or row.parent_output_id is not None
            or row.attempt != 0 or row.error_code or row.item.status != "success" or row.item.attempts != 0
            or not row.item.manifest_complete or row.item.batch.collection_id != ref.collection_id
            or row.item.batch.user_id != retained.get("actor_id")
            or row.item.source_file.name != payload["source_artifact"]["storage_key"]
            or row.item.file_size_bytes != payload["source_artifact"]["size_bytes"]
            or row.item.original_filename != posixpath.basename(evidence["attached_reference"]["name"])
            or row.item.content_type != "application/pdf"):
        _unavailable()
    return OutputResultV1(row.output_key, "success", ref, None)


def _adopt(ref, actor_id, *, using, deadline):
    first = _current(ref, actor_id, using=using)
    owner, chain, (storage, name, _) = first
    origins = _origins(ref, using=using)
    if len(origins) > 1 or (origins and origins[0].provenance_kind != "current_adoption"):
        _unavailable()
    filename = posixpath.basename(name)
    if not filename or len(filename) > 300:
        _unavailable()
    policy = pdf_current_validation.freeze_policy()
    _remaining(deadline)
    captured = source_storage.capture_attached_file(storage, name, deadline=deadline)
    canonical = owner.full_text.encode("utf-8")
    receipt = pdf_current_validation.validate(policy, captured.data, canonical, ref.source_hash, deadline)
    if (receipt.get("status") != "match" or receipt.get("input_sha256") != captured.sha256
            or receipt.get("input_bytes") != captured.size_bytes
            or receipt.get("projected_sha256") != ref.source_hash
            or receipt.get("projected_bytes") != len(canonical)
            or receipt.get("projected_codepoints") != len(owner.full_text)):
        _unavailable()
    _same_current(first, _current(ref, actor_id, using=using))
    artifact = source_artifacts.persist_artifact(captured.data, member_path=None, container_sha256=None,
        using=using, deadline=deadline)
    occurrence = source_artifacts.persist_occurrence(artifact, original_filename=filename,
        content_type="application/pdf", archive_lineage=(), using=using, deadline=deadline)
    extraction = ExtractionV1("current-validation:" + policy["key"], artifact, owner.full_text,
        (SourceElementV1("body", "paragraph", 0, len(owner.full_text), None, None, None, None),),
        "partial", ("structure_unavailable",))
    revision_id = extraction_revisions.persist_extraction_revision(occurrence, extraction,
        using=using, deadline=deadline)
    revision = ExtractionRevision.objects.using(using).get(pk=revision_id)
    payload = _payload(owner, chain, artifact)

    # Native collection advisory locks/rows precede the native document UUID
    # advisory lock and bounded exact PDF row lock. No source I/O occurs here.
    with locked_collection_parent_rows(ref.collection_id, [pk for pk, _ in chain], using=using):
        _lock_document_scope(ref.id, using=using)
        _same_current(first, _current(ref, actor_id, using=using, locked=True))
        with connections[using].cursor() as cursor:
            cursor.execute("SELECT clock_timestamp()")
            validated_at = cursor.fetchone()[0]
        evidence = {"version": "current-adoption-evidence-v1", "policy": policy,
            "document": document_payload(ref), "attached_reference": {"field": "pdf_file", "name": name},
            "artifact": {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes},
            "occurrence_key": occurrence.occurrence_key,
            "extraction_revision": {"id": revision.pk, "manifest_sha256": revision.manifest_sha256},
            "actor_id": actor_id, "validated_at": validated_at.isoformat()}
        origins = _origins(ref, using=using)
        if origins:
            if len(origins) != 1:
                _unavailable()
            return _identical(origins[0], ref, payload, evidence, occurrence, revision)
        batch = IngestionBatch.objects.using(using).create(user_id=actor_id, collection_id=ref.collection_id)
        item = IngestionBatchItem.objects.using(using).create(batch=batch, source_file=artifact.storage_key,
            original_filename=filename, content_type="application/pdf", file_size_bytes=artifact.size_bytes,
            status="success", attempts=0, manifest_complete=False)
        key = output_key(item.pk, occurrence.occurrence_key, extraction.parser_key, "text", "body")
        IngestionOutput.objects.using(using).create(item=item, output_key=key,
            source_occurrence_id=occurrence.occurrence_id, extraction_revision_id=revision.pk,
            output_kind="text", element_key="body", payload=payload, status="success",
            provenance_kind="current_adoption", provenance_evidence=evidence,
            document_model=ref.model_label, document_pkid=ref.pkid, document_id=ref.id,
            document_collection_id=ref.collection_id, document_source_hash=ref.source_hash)
        _, item.document_ids, item.parser_metadata = item_snapshot(item, using=using, validate_scope=False)
        item.parser_metadata["provenance_kind"] = "current_adoption"
        item.manifest_complete, item.finished_at = True, validated_at
        # Extraction remains partial; the current-validation operation succeeded.
        item.save(using=using, update_fields=["document_ids", "parser_metadata", "manifest_complete", "finished_at"])
        _remaining(deadline)
    return OutputResultV1(key, "success", ref, None)


def adopt_current_pdf(document: DocumentRefV1, *, actor_id: int, using: str,
                      deadline: float | None = None) -> OutputResultV1:
    """Validate current attached bytes and atomically bind their existing owner.

    Accepts no source, parser, policy, receipt or parity input. Exact repeats
    revalidate current authority/bytes and reuse the original output unchanged.
    """
    document_payload(document)
    if (document.model_label != PDFDocument._meta.label_lower or type(actor_id) is not int
            or not 0 < actor_id <= 9_223_372_036_854_775_807 or type(using) is not str
            or not using or using not in connections
            or (deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)))):
        _unavailable()
    deadline = min(time.monotonic() + 120, deadline) if deadline is not None else time.monotonic() + 120
    _remaining(deadline)
    connection = connections[using]
    # This internal operator operation owns its final transaction. An enclosing
    # caller transaction could hold locks across capture or roll back its result.
    if connection.vendor != "postgresql" or connection.in_atomic_block or not connection.get_autocommit():
        _unavailable()
    try:
        with _bounded_database(connection, deadline):
            return _adopt(document, actor_id, using=using, deadline=deadline)
    except SourceScopeUnavailable:
        raise
    except (DatabaseError, source_artifacts.SourceArtifactIntegrityError,
            source_artifacts.UnsupportedArtifactStorage, ValueError, RuntimeError) as exc:
        raise SourceScopeUnavailable("source_scope_unavailable") from exc
