import re

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestNavbar:
    """Top navigation bar behaviour."""

    def test_navbar_is_visible(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Navbar is visible on every authenticated page."""
        page.goto(f"{live_server.url}/")
        expect(page.locator("nav")).to_be_visible()

    def test_navbar_logo_links_to_home(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking the navbar logo navigates to the homepage."""
        page.goto(f"{live_server.url}/aquillm/search/")
        page.locator("#nav-logo").click()
        page.wait_for_load_state("networkidle")
        assert page.url.rstrip("/") == live_server.url.rstrip("/") or page.url == f"{live_server.url}/"

    def test_navbar_shows_aquillm_brand(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Navbar displays AquiLLM branding text."""
        page.goto(f"{live_server.url}/")
        expect(page.locator("nav").get_by_text("AquiLLM")).to_be_visible()


class TestSidebar:
    """Sidebar navigation behaviour."""

    def test_sidebar_visible_by_default(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Sidebar is visible when the page first loads."""
        page.goto(f"{live_server.url}/")
        expect(page.locator("#aq-sidebar")).to_be_visible()

    def test_sidebar_toggle_collapses(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking the menu toggle collapses the sidebar."""
        page.goto(f"{live_server.url}/")
        page.locator("#menu-toggle").click()
        expect(page.locator("#aq-sidebar")).to_have_attribute(
            "class", re.compile(r"ml-\[-260px\]")
        )

    def test_sidebar_toggle_expands(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking the menu toggle twice re-expands the sidebar."""
        page.goto(f"{live_server.url}/")
        toggle = page.locator("#menu-toggle")
        toggle.click()
        toggle.click()
        expect(page.locator("#aq-sidebar")).not_to_have_attribute(
            "class", re.compile(r"ml-\[-260px\]")
        )

    def test_sidebar_has_menu_header(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Sidebar has a Menu header."""
        page.goto(f"{live_server.url}/")
        expect(page.locator("#sidebar-header span")).to_have_text("Menu")

    def test_sidebar_has_utilities_section(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Sidebar includes a Utilities section."""
        page.goto(f"{live_server.url}/")
        expect(page.locator("#aq-sidebar")).to_contain_text("Utilities")

    def test_floating_logo_appears_when_sidebar_collapsed(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """When sidebar is collapsed the floating logo becomes visible."""
        page.goto(f"{live_server.url}/")
        page.locator("#menu-toggle").click()
        # Give the transition time to complete
        page.wait_for_timeout(350)
        floating = page.locator("#floating-logo")
        expect(floating).to_be_visible()


class TestAccountMenu:
    """Account management dropdown behaviour."""

    def test_account_menu_opens(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking the account button opens the account menu."""
        page.goto(f"{live_server.url}/")
        page.locator("#account-management-toggle-button").click()
        expect(page.locator("#account-menu-modal")).to_be_visible()

    def test_account_menu_shows_manage_account(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Account menu has a Manage Account option."""
        page.goto(f"{live_server.url}/")
        page.locator("#account-management-toggle-button").click()
        expect(
            page.locator("#account-menu-modal").get_by_text("Manage Account")
        ).to_be_visible()

    def test_account_menu_shows_logout(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Account menu has a Logout button."""
        page.goto(f"{live_server.url}/")
        page.locator("#account-management-toggle-button").click()
        expect(
            page.locator("#account-menu-modal").get_by_role("button", name="Logout")
        ).to_be_visible()

    def test_account_menu_closes_on_second_click(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking the account button again closes the menu."""
        page.goto(f"{live_server.url}/")
        btn = page.locator("#account-management-toggle-button")
        btn.click()
        expect(page.locator("#account-menu-modal")).to_be_visible()
        btn.click()
        expect(page.locator("#account-menu-modal")).not_to_be_visible()


class TestAuthenticatedHomepage:
    """Homepage content for authenticated users."""

    def test_authenticated_homepage_shows_welcome(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Authenticated users see the welcome heading."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-heading")).to_contain_text("Welcome to AquiLLM")

    def test_authenticated_homepage_shows_subtext(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Authenticated users see the tagline subtext."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-subtext").first).to_contain_text(
            "Preserving Tacit Knowledge"
        )

    def test_authenticated_homepage_no_login_form(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Authenticated users do not see the login prompt."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-login-link")).not_to_be_visible()

    def test_authenticated_homepage_has_logo_badge(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Authenticated homepage shows the logo badge."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-logo-badge")).to_be_visible()
