"""Public source offers must be usable without an account or database."""

from django.test import SimpleTestCase, override_settings


class LicensingTests(SimpleTestCase):
    @override_settings(AQUILLM_SOURCE_URL="https://example.org/source/release-123.tar.gz")
    def test_anonymous_source_offer_links_to_the_operators_deployed_source(self):
        response = self.client.get("/legal/")
        self.assertContains(response, "https://example.org/source/release-123.tar.gz")
        self.assertContains(response, "Source for this deployment")
        self.assertContains(response, "AGPL")
        self.assertContains(response, "THIRD_PARTY_LICENSES.txt")

    def test_unconfigured_offer_does_not_present_upstream_as_deployed_source(self):
        response = self.client.get("/legal/")
        self.assertContains(response, "Upstream repository")
        self.assertNotContains(response, "Source for this deployment")
        self.assertContains(response, "has not configured")

    def test_unsafe_source_urls_are_not_rendered_as_links(self):
        for url in ("javascript:alert(1)", "//example.org", "https://user:secret@example.org/src",
                    "https://example.org:abc/source.zip", "https://example.org:999999/source.zip"):
            with self.subTest(url=url), override_settings(AQUILLM_SOURCE_URL=url):
                response = self.client.get("/legal/")
                self.assertNotContains(response, url)
                self.assertNotContains(response, "Source for this deployment")

    def test_license_is_available_as_plain_text_without_login(self):
        response = self.client.get("/legal/project-license/")
        self.assertContains(response, "Version 2, June 1991")
        self.assertEqual(response["Content-Type"], "text/plain; charset=utf-8")

    def test_notice_route_cannot_read_arbitrary_files(self):
        self.assertEqual(self.client.get("/legal/documents/.env/").status_code, 404)
        self.assertContains(self.client.get("/legal/third-party/"), "PyMuPDF")

    def test_deployment_check_rejects_missing_or_unsafe_source_offer(self):
        from django.core.checks import run_checks

        from apps.core.apps import CoreConfig

        CoreConfig("apps.core", __import__("apps.core", fromlist=["core"])).ready()

        for value in ("", "javascript:alert(1)", "https://example.org:abc/source.zip"):
            with self.subTest(value=value), override_settings(AQUILLM_SOURCE_URL=value):
                errors = run_checks(tags=["licensing"], include_deployment_checks=True)
                self.assertEqual([e.id for e in errors], ["licensing.E001"])
        with override_settings(AQUILLM_SOURCE_URL="https://example.org/source/v1.zip"):
            self.assertEqual(run_checks(tags=["licensing"], include_deployment_checks=True), [])
