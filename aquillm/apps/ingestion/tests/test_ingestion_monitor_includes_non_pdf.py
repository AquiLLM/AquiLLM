import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from aquillm.models import Collection, CollectionPermission, RawTextDocument


@pytest.mark.django_db
def test_ingestion_monitor_includes_non_pdf_documents(client):
    user = User.objects.create_user(username="monitor-user", password="pw12345")
    collection = Collection.objects.create(name="Monitor Collection")
    CollectionPermission.objects.create(user=user, collection=collection, permission="EDIT")

    doc = RawTextDocument.objects.create(
        title="Image OCR Result",
        full_text="Example extracted text",
        collection=collection,
        ingested_by=user,
        ingestion_complete=False,
    )

    client.force_login(user)
    response = client.get(reverse("api_ingestion_monitor"))
    assert response.status_code == 200
    payload = response.json()
    matched = next(item for item in payload if item["documentId"] == str(doc.id))
    assert matched["modality"] == "text"
    assert matched["textExtracted"] is True
