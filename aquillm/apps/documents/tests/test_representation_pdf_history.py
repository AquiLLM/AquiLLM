"""Real native PDF ingestion retains historical bytes under current permission."""
import json

import fitz
import pytest
from django.apps import apps
from django.core.files.storage import default_storage
from django.test import Client, RequestFactory

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.views.api import chunk_detail
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .representation_support import upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


def test_native_pdf_old_pk_retains_artifact_after_reparse_move_and_current_path_change(upload):
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Original native PDF table value 10")
        data = pdf.tobytes()
    item = upload((("original.pdf", data),))
    run_ingest_uploaded_file(item.pk)
    output = item.outputs.get(output_kind="text", status="success")
    doc = apps.get_model(output.document_model).objects.get(pk=output.document_pkid)
    old = TextChunk.objects.bulk_create([TextChunk(doc_id=doc.id, content=doc.full_text,
        start_position=0, end_position=len(doc.full_text), chunk_number=0)])[0]
    assert activate(doc, output, text="Corrected PDF table value 100") == "activated"
    request = RequestFactory().get("/chunk/", {"include_full_text": "0"})
    request.user = doc.ingested_by
    payload = json.loads(chunk_detail(request, old.pk).content)
    assert payload["document"]["has_pdf"] and payload["source_context_status"] == "verified"
    client = Client()
    client.force_login(doc.ingested_by)
    from django.urls import reverse
    url = reverse("pdf", kwargs={"doc_id": doc.id}) + f"?chunk_id={old.pk}"
    response = client.get(url)
    assert response.status_code == 200 and b"".join(response.streaming_content) == data
    doc.refresh_from_db()
    if hasattr(doc, "pdf_file"):
        doc.pdf_file = "unrelated-current.pdf"
        doc.save(dont_rechunk=True, update_fields=["pdf_file"])
    target = Collection.objects.create(name="Moved PDF current permission")
    permission = CollectionPermission.objects.create(user=doc.ingested_by, collection=target, permission="EDIT")
    doc.move_to(target, actor=doc.ingested_by)
    response = client.get(url)
    assert response.status_code == 200 and b"".join(response.streaming_content) == data
    permission.delete()
    assert client.get(url).status_code == 403
    CollectionPermission.objects.create(user=doc.ingested_by, collection=target, permission="VIEW")
    key = output.payload["source_artifact"]["storage_key"]
    with default_storage.open(key, "wb") as stream:
        stream.write(b"corrupt immutable PDF")
    assert client.get(url).status_code == 404
