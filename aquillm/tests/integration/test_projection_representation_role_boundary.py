"""A restricted projection role validates persisted sources without body grants."""
from uuid import uuid4

import pytest
from django.db import ProgrammingError, connections, transaction

from apps.documents.models import RawTextDocument
from apps.knowledge_graph.projection.worker import project_generation
from tests.integration.projection_boundary_support import (
    boundary_projection, claim, graph_configuration, graph_source, projection_roles,
    source_call, upload,
)

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def test_real_role_worker_publishes_actual_nonnull_source(boundary_projection):
    source, row = boundary_projection
    outcome = project_generation(projection_id=row.pk, lease_owner="boundary-worker")
    assert outcome.ready, outcome.failure_code
    row.refresh_from_db()
    assert row.state == "ready"
    assert source_call(row, purpose="audit", lease_owner=None) is True
    fields, rows = source_call(row, purpose="audit", lease_owner=None, coordinates=True)
    assert fields == ("id", "doc_id", "chunk_number")
    assert rows == list(source.document.chunks.order_by("pk").values_list("pk", "doc_id", "chunk_number"))


@pytest.mark.parametrize("change", [
    {"projection_id": uuid4()}, {"generation_key": uuid4()},
    {"representation_digest": "f" * 64}, {"lease_owner": "foreign"},
    {"lease_owner": None}, {"purpose": "pending"}, {"purpose": "audit", "lease_owner": None},
])
def test_exact_projection_and_original_claim_are_required(boundary_projection, change):
    _, row = boundary_projection
    claim(row)
    assert source_call(row, **change) is False


def test_pending_projection_has_no_read_admission(boundary_projection):
    _, row = boundary_projection
    assert source_call(row) is False


@pytest.mark.parametrize("kind", ["pdf", "figure"])
def test_cross_model_uuid_collision_revokes_original_binding(boundary_projection, kind):
    from apps.documents.models import DocumentFigure, PDFDocument

    source, row = boundary_projection
    claim(row)
    assert source_call(row) is True
    # Deliberate malformed import: second concrete owner, no graph/source rewrite.
    model = PDFDocument if kind == "pdf" else DocumentFigure
    collision = model(
        id=source.document.id, title="collision", collection_id=source.document.collection_id,
        ingested_by_id=source.document.ingested_by_id, full_text=source.document.full_text,
        full_text_hash=source.document.full_text_hash)
    if kind == "figure":
        collision.parent_document = source.document
        collision.source_format = "txt"
    model.objects.bulk_create([collision])
    assert source_call(row) is False


@pytest.mark.parametrize("change", ["body", "move", "delete", "generation"])
def test_current_source_change_revokes_actual_role_read(boundary_projection, change):
    source, row = boundary_projection
    claim(row)
    assert source_call(row) is True
    if change == "body":
        RawTextDocument.objects.filter(pk=source.document.pk).update(full_text="changed body")
    elif change == "move":
        from apps.collections.models import Collection
        other = Collection.objects.create(name="projection boundary destination")
        source.document.collection = other
        source.document.save(dont_rechunk=True)
    elif change == "delete":
        source.document.delete()
    else:
        source.activate_only("Orion uses MMLU. Replacement observation.")
    assert source_call(row) is False


@pytest.mark.parametrize("alias", ["projection_source", "projection_state"])
def test_private_validator_is_not_granted(boundary_projection, alias):
    _, row = boundary_projection
    with pytest.raises(ProgrammingError, match="permission denied"):
        with transaction.atomic(using=alias), connections[alias].cursor() as cursor:
            cursor.execute("SELECT public.kg_projection_artifact_representation_current_v1(%s)",
                           [row.artifact_id])


def test_state_role_cannot_call_source_api_and_public_has_no_function_grant(boundary_projection):
    _, row = boundary_projection
    with pytest.raises(ProgrammingError, match="permission denied"):
        with transaction.atomic(using="projection_state"), connections["projection_state"].cursor() as cursor:
            cursor.execute("SELECT public.kg_projection_representation_current_v1(%s,%s,%s,%s,%s)",
                           [row.pk, row.generation_key, row.representation_digest, "build", "boundary-worker"])
    with connections["default"].cursor() as cursor:
        cursor.execute("SELECT p.proname FROM pg_proc p CROSS JOIN LATERAL "
                       "aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a "
                       "WHERE p.proname IN (%s,%s,%s,%s) AND a.grantee=0 AND a.privilege_type='EXECUTE'",
                       ["kg_projection_representation_current_v1", "kg_projection_representation_chunks_v1",
                        "kg_projection_artifact_representation_current_v1", "kg_projection_payload_valid_v1"])
        assert cursor.fetchall() == []


def test_source_role_keeps_concrete_and_representation_bodies_private(boundary_projection):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    for table in [model._meta.db_table for model in DESCENDED_FROM_DOCUMENT] + [
        "aquillm_representationgeneration", "aquillm_legacysourcesnapshot"]:
        field = "full_text" if table not in {
            "aquillm_representationgeneration", "aquillm_legacysourcesnapshot"} else "canonical_text"
        with pytest.raises(ProgrammingError, match="permission denied"):
            with transaction.atomic(using="projection_source"), connections["projection_source"].cursor() as cursor:
                cursor.execute(f'SELECT "{field}" FROM public."{table}" LIMIT 1')
