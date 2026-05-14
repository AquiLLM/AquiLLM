import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestUserSettingsPage:
    """User settings page behaviour."""

    def test_settings_page_loads(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """User settings page loads for authenticated users."""
        page.goto(f"{live_server.url}/user-settings/")
        page.wait_for_load_state("networkidle")
        assert "/accounts/login/" not in page.url

    def test_settings_page_has_react_mount(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Settings page contains the React mount point."""
        page.goto(f"{live_server.url}/user-settings/")
        expect(page.locator("#user-settings-root")).to_be_visible()

    def test_settings_page_has_sidebar(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Settings page includes the sidebar navigation."""
        page.goto(f"{live_server.url}/user-settings/")
        expect(page.locator("#aq-sidebar")).to_be_visible()

    def test_settings_page_has_navbar(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Settings page includes the top navigation bar."""
        page.goto(f"{live_server.url}/user-settings/")
        expect(page.locator("#nav-logo")).to_be_visible()
