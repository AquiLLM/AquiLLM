"""Permission tests for the feedback dashboard page."""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

User = get_user_model()


def make_superuser(username: str = "pageadmin") -> User:
    return User.objects.create_superuser(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


def make_staff_user(username: str = "pagestaff") -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
        is_staff=True,
    )


def make_regular_user(username: str = "pageregular") -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


class FeedbackDashboardPagePermissionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.superuser = make_superuser()
        self.staff_user = make_staff_user()
        self.regular_user = make_regular_user()
        self.url = reverse("feedback_dashboard")

    def test_unauthenticated_user_redirects_to_login(self):
        resp = self.client.get(self.url)

        self.assertIn(resp.status_code, [301, 302])
        self.assertIn("login", resp["Location"].lower())

    def test_regular_user_is_forbidden(self):
        self.client.login(username="pageregular", password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_is_forbidden(self):
        self.client.login(username="pagestaff", password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)

    def test_superuser_gets_page(self):
        self.client.login(username="pageadmin", password="testpass123")

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_superuser_page_uses_dashboard_template(self):
        self.client.login(username="pageadmin", password="testpass123")

        resp = self.client.get(self.url)
        self.assertTemplateUsed(resp, "aquillm/feedback_dashboard.html")

    def test_superuser_page_extends_base_template(self):
        self.client.login(username="pageadmin", password="testpass123")

        resp = self.client.get(self.url)
        self.assertTemplateUsed(resp, "aquillm/base.html")
