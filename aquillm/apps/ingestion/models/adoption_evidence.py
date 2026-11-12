"""Closed present-validation evidence; this does not attest historical ingestion.

The eventual trusted adoption service must generate these identities from its
own policy execution and captured bytes. Shape checks are not a parity proof.
"""
import re
from datetime import datetime

from django.core.exceptions import ValidationError
from django.utils import timezone


def validate_adoption_evidence(row):
    evidence = row.provenance_evidence
    if row.provenance_kind == "ingestion":
        if evidence != {}:
            raise ValidationError("Ingestion cannot carry current-adoption evidence.")
        return
    if row.provenance_kind != "current_adoption":
        raise ValidationError("Invalid output provenance kind.")

    def closed(value, keys):
        return type(value) is dict and set(value) == set(keys.split())

    def digest(value):
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    valid = (
        row.status == "success" and row.output_kind == "text" and row.element_key == "body"
        and row.parent_output_id is None and row.attempt == 0 and row.error_code == ""
        and row.document_model == "apps_documents.pdfdocument"
        and closed(evidence, "version policy document attached_reference artifact occurrence_key extraction_revision actor_id validated_at")
    )
    if valid:
        policy = evidence["policy"]
        valid = (
            evidence["version"] == "current-adoption-evidence-v1"
            and closed(policy, "key version implementation_sha256 runtime_sha256")
            and policy.get("key") == "legacy-upload-PDF-current-validation-v1"
            and policy.get("version") == "1"
            and digest(policy.get("implementation_sha256")) and digest(policy.get("runtime_sha256"))
            and evidence["document"] == {
                "model_label": row.document_model, "pkid": row.document_pkid,
                "id": str(row.document_id), "collection_id": row.document_collection_id,
                "source_hash": row.document_source_hash,
            }
            and closed(evidence["attached_reference"], "field name")
            and evidence["attached_reference"].get("field") == "pdf_file"
            and type(evidence["attached_reference"].get("name")) is str
            and 0 < len(evidence["attached_reference"]["name"]) <= 500
            and closed(evidence["artifact"], "sha256 size_bytes")
            and digest(evidence["artifact"].get("sha256"))
            and type(evidence["artifact"].get("size_bytes")) is int
            and evidence["artifact"]["size_bytes"] > 0
            and digest(evidence["occurrence_key"])
            and closed(evidence["extraction_revision"], "id manifest_sha256")
            and type(evidence["extraction_revision"].get("id")) is int
            and evidence["extraction_revision"]["id"] == row.extraction_revision_id
            and digest(evidence["extraction_revision"].get("manifest_sha256"))
            and type(evidence["actor_id"]) is int and evidence["actor_id"] > 0
        )
    if valid:
        try:
            recorded = evidence["validated_at"]
            valid = type(recorded) is str and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})", recorded) is not None
            # Operation metadata is not a bearer authorization or freshness proof.
            valid = valid and timezone.is_aware(datetime.fromisoformat(recorded))
        except (TypeError, ValueError):
            valid = False
    if not valid:
        raise ValidationError("Invalid current-adoption evidence.")
