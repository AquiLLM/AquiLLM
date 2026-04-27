"""
permission tests for the feedback dashboard page and all four api endpoints

covers:
    unauthenticated users redirected to login on page and all apis
    regular users get 403 on page and all apis
    staff users (is_staff but not is_superuser) get 403 on page and all apis
    superusers get 200 on page and non-403 on all apis
"""
from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

User = get_user_model()


class FeedbackDashboardPermissionTests(TestCase):

    def setUp(self):
        self.client = Client()

        self.superuser = User.objects.create_superuser(
            username="superadmin",
            email="superadmin@test.com",
            password="testpass123",
        )
        self.staff_user = User.objects.create_user(
            username="staffuser",
            email="staff@test.com",
            password="testpass123",
            is_staff=True,
        )
        self.regular_user = User.objects.create_user(
            username="regularuser",
            email="regular@test.com",
            password="testpass123",
        )

        self.page_url = reverse("feedback_dashboard")

        # all four dashboard api endpoints must enforce superuser access
        self.api_urls = [
            reverse("api_feedback_dashboard_rows"),
            reverse("api_feedback_dashboard_summary"),
            reverse("api_feedback_dashboard_filters"),
            reverse("api_feedback_dashboard_export"),
        ]


    def test_unauthenticated_page_redirects_to_login(self):
        resp = self.client.get(self.page_url)
        self.assertIn(resp.status_code, [301, 302])
        self.assertIn("login", resp["Location"].lower())

    def test_unauthenticated_api_rows_redirects(self):
        resp = self.client.get(reverse("api_feedback_dashboard_rows"))
        self.assertIn(resp.status_code, [301, 302])

    def test_unauthenticated_api_summary_redirects(self):
        resp = self.client.get(reverse("api_feedback_dashboard_summary"))
        self.assertIn(resp.status_code, [301, 302])

    def test_unauthenticated_api_filters_redirects(self):
        resp = self.client.get(reverse("api_feedback_dashboard_filters"))
        self.assertIn(resp.status_code, [301, 302])

    def test_unauthenticated_api_export_redirects(self):
        resp = self.client.get(reverse("api_feedback_dashboard_export"))
        self.assertIn(resp.status_code, [301, 302])


    def test_regular_user_page_forbidden(self):
        self.client.login(username="regularuser", password="testpass123")
        resp = self.client.get(self.page_url)
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_api_rows_forbidden(self):
        self.client.login(username="regularuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_rows"))
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_api_summary_forbidden(self):
        self.client.login(username="regularuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_summary"))
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_api_filters_forbidden(self):
        self.client.login(username="regularuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_filters"))
        self.assertEqual(resp.status_code, 403)

    def test_regular_user_api_export_forbidden(self):
        self.client.login(username="regularuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_export"))
        self.assertEqual(resp.status_code, 403)

    # ------------------------------------------------------------------
    # staff user (is_staff=True but is_superuser=False)
    # ------------------------------------------------------------------

    def test_staff_user_page_forbidden(self):
        self.client.login(username="staffuser", password="testpass123")
        resp = self.client.get(self.page_url)
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_api_rows_forbidden(self):
        self.client.login(username="staffuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_rows"))
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_api_summary_forbidden(self):
        self.client.login(username="staffuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_summary"))
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_api_filters_forbidden(self):
        self.client.login(username="staffuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_filters"))
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_api_export_forbidden(self):
        self.client.login(username="staffuser", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_export"))
        self.assertEqual(resp.status_code, 403)

    # ------------------------------------------------------------------
    # superuser
    # ------------------------------------------------------------------

    def test_superuser_page_returns_200(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(self.page_url)
        self.assertEqual(resp.status_code, 200)

    def test_superuser_page_uses_correct_template(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(self.page_url)
        self.assertTemplateUsed(resp, "aquillm/feedback_dashboard.html")

    def test_superuser_page_extends_base(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(self.page_url)
        self.assertTemplateUsed(resp, "aquillm/base.html")

    def test_superuser_api_rows_not_forbidden(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_rows"))
        self.assertNotEqual(resp.status_code, 403)

    def test_superuser_api_summary_returns_200(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_summary"))
        self.assertEqual(resp.status_code, 200)

    def test_superuser_api_filters_returns_200(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_filters"))
        self.assertEqual(resp.status_code, 200)

    def test_superuser_api_export_not_forbidden(self):
        self.client.login(username="superadmin", password="testpass123")
        resp = self.client.get(reverse("api_feedback_dashboard_export"))
        self.assertNotEqual(resp.status_code, 403)