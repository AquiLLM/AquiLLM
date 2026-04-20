"""RestrictDomains.list_apps deduplicates DB + settings Google apps (allauth merge)."""

from unittest.mock import MagicMock, patch

from django.test import RequestFactory, SimpleTestCase

from aquillm.adapters import RestrictDomains


class TestRestrictDomainsListAppsDedupe(SimpleTestCase):
    def test_deduplicates_same_provider_and_client_id(self):
        request = RequestFactory().get("/")
        first = MagicMock()
        first.provider = "google"
        first.provider_id = "google"
        first.client_id = "same-oauth-client"
        duplicate = MagicMock()
        duplicate.provider = "google"
        duplicate.provider_id = "google"
        duplicate.client_id = "same-oauth-client"
        with patch(
            "allauth.socialaccount.adapter.DefaultSocialAccountAdapter.list_apps",
            return_value=[first, duplicate],
        ):
            out = RestrictDomains().list_apps(request)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], first)

    def test_keeps_distinct_client_ids(self):
        request = RequestFactory().get("/")
        a = MagicMock()
        a.provider = "google"
        a.provider_id = "google"
        a.client_id = "client-a"
        b = MagicMock()
        b.provider = "google"
        b.provider_id = "google"
        b.client_id = "client-b"
        with patch(
            "allauth.socialaccount.adapter.DefaultSocialAccountAdapter.list_apps",
            return_value=[a, b],
        ):
            out = RestrictDomains().list_apps(request)
        self.assertEqual(len(out), 2)
