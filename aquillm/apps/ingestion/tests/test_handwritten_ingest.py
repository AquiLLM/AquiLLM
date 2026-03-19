from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from aquillm.models import Collection, CollectionPermission


class HandwrittenIngestAjaxTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="tester", password="pw12345")
        self.client.force_login(self.user)
        self.collection = Collection.objects.create(name="Test Collection")
        CollectionPermission.objects.create(
            user=self.user,
            collection=self.collection,
            permission="EDIT",
        )

    def test_invalid_ajax_request_returns_field_level_error_details(self):
        response = self.client.post(
            reverse("ingest_handwritten_notes"),
            data={
                "title": "Bad Upload",
                "collection": str(self.collection.id),
            },
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertIn("image_file", payload["error"])
