"""Bounded metadata planning for exact persisted representation inputs."""
from dataclasses import dataclass

from apps.documents.models import DESCENDED_FROM_DOCUMENT, DocumentRepresentationState
from apps.ingestion.models import ExtractionRevision, IngestionOutput
from .representation_caption import caption_chunker_key
from .representation_contracts import SourceScopeUnavailable
from .representation_history import reference
from .representation_inputs import configured_chunker_key
from .representation_snapshot_build import snapshot_chunker_key


@dataclass(frozen=True)
class RepresentationPlan:
    document: object
    parser_key: str
    chunker_key: str
    revision_id: int | None
    status: str

    def report(self, *, using):
        ref = self.document
        active = DocumentRepresentationState.objects.using(using).filter(
            model_label=ref.model_label, document_pkid=ref.pkid, document_id=ref.id,
        ).values_list("active_generation_id", flat=True).first()
        return dict(model_label=ref.model_label, document_pkid=ref.pkid,
            document_id=str(ref.id), collection_id=ref.collection_id,
            source_hash=ref.source_hash, current_generation=str(active) if active else None,
            parser_key=self.parser_key, chunker_key=self.chunker_key,
            extraction_revision_id=self.revision_id, status=self.status)


def registered_chunker_keys():
    return (configured_chunker_key(), caption_chunker_key(), snapshot_chunker_key())


def plan_document(document, *, parser_key, chunker_key, revision_id, using):
    """Never load canonical bodies, artifact bytes, or revision element payloads."""
    ref = reference(document)
    def result(status, revision=revision_id):
        return RepresentationPlan(ref, parser_key, chunker_key, revision, status)
    owners = []
    for model in DESCENDED_FROM_DOCUMENT:
        owners.extend((model._meta.label_lower, pk) for pk in model._base_manager.using(using)
                      .filter(id=ref.id).values_list("pkid", flat=True)[:2])
    if owners != [(ref.model_label, ref.pkid)]:
        return result("ambiguous_owner")
    outputs = list(IngestionOutput.objects.using(using).filter(
        document_model=ref.model_label, document_pkid=ref.pkid, document_id=ref.id,
    ).values("pk", "status", "output_kind", "source_occurrence_id", "extraction_revision_id")[:2])
    if not outputs:
        if (ref.model_label == "apps_documents.rawtextdocument" and revision_id is None
            and parser_key == "canonical-snapshot-v1" and chunker_key == snapshot_chunker_key()):
            return result("planned")
        return result("durable_input_unavailable")
    if len(outputs) != 1 or outputs[0]["status"] != "success":
        return result("ambiguous_output")
    output = outputs[0]
    expected_chunker = configured_chunker_key() if output["output_kind"] == "text" else caption_chunker_key()
    if chunker_key != expected_chunker:
        return result("chunker_policy_mismatch")
    revisions = ExtractionRevision.objects.using(using).filter(
        occurrence_id=output["source_occurrence_id"], parser_key=parser_key,
    ).exclude(quality="unsupported")
    if revision_id is not None:
        revisions = revisions.filter(pk=revision_id)
    ids = list(revisions.order_by("pk").values_list("pk", flat=True)[:2])
    if len(ids) != 1:
        return result("ambiguous_revision" if ids else "revision_unavailable")
    if output["output_kind"] != "text" and ids[0] != output["extraction_revision_id"]:
        return result("output_revision_mismatch")
    return result("planned", ids[0])


def execute_plan(plan, *, using):
    if plan.status != "planned":
        raise SourceScopeUnavailable("representation_plan_unavailable")
    if plan.revision_id is None:
        from .representation_snapshot_build import build_canonical_snapshot_representation
        return build_canonical_snapshot_representation(plan.document, using=using)
    from .representation_activation import build_document_representation
    return build_document_representation(plan.document, extraction_revision_id=plan.revision_id,
                                         chunker_key=plan.chunker_key, using=using)
