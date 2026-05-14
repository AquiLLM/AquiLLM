import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestUnauthenticatedRedirects:
    """Unauthenticated users are redirected to login for protected pages."""

    PROTECTED_PATHS = [
        "/aquillm/search/",
        "/aquillm/user_collections/",
        "/aquillm/user_ws_convos/",
        "/new_ws_convo/",
    ]

    @pytest.mark.parametrize("path", PROTECTED_PATHS)
    def test_protected_page_redirects_to_login(
        self, e2e_settings, live_server, page, path
    ):
        """Visiting a protected page without auth redirects to /accounts/login/."""
        page.goto(f"{live_server.url}{path}")
        page.wait_for_load_state("networkidle")
        assert "/accounts/login/" in page.url


class TestUnauthenticatedHomepage:
    """Homepage behaviour for unauthenticated visitors."""

    def test_homepage_shows_login_prompt(self, e2e_settings, live_server, page):
        """Unauthenticated visitors see a login prompt on the homepage."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-login-link")).to_be_visible()
        expect(page.locator(".welcome-heading")).to_contain_text("Welcome to AquiLLM")

    def test_homepage_login_prompt_submits_to_login(
        self, e2e_settings, live_server, page
    ):
        """The login prompt form submits to the allauth login URL."""
        page.goto(f"{live_server.url}/")
        form = page.locator(".welcome-shell form[action*='login']")
        expect(form).to_be_visible()

    def test_homepage_shows_welcome_subtext(self, e2e_settings, live_server, page):
        """Homepage shows the project tagline."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-subtext").first).to_contain_text(
            "Preserving Tacit Knowledge"
        )

    def test_homepage_shows_logo_badge(self, e2e_settings, live_server, page):
        """Homepage shows the AquiLLM logo badge."""
        page.goto(f"{live_server.url}/")
        expect(page.locator(".welcome-logo-badge")).to_be_visible()
        expect(page.locator(".welcome-logo")).to_be_visible()


class TestLogout:
    """Logout flow works correctly."""

    def test_logout_via_account_menu(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking logout in account menu logs the user out."""
        page.goto(f"{live_server.url}/")
        page.locator("#account-management-toggle-button").click()
        menu = page.locator("#account-menu-modal")
        expect(menu).to_be_visible()
        menu.get_by_role("button", name="Logout").click()
        page.wait_for_load_state("networkidle")
        # After logout, visiting a protected page should redirect to login
        page.goto(f"{live_server.url}/aquillm/search/")
        page.wait_for_load_state("networkidle")
        assert "/accounts/login/" in page.url
