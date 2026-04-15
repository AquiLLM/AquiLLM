import re
import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


def test_homepage_loads_when_authenticated(
    e2e_settings, live_server, page, authenticated_user
):
    """Authenticated user can access the homepage."""
    page.goto(f"{live_server.url}/")
    # Should not be redirected to login
    assert "/accounts/login/" not in page.url
    # Navbar should be visible with AquiLLM branding
    expect(page.locator("#nav-logo")).to_be_visible()


def test_sidebar_opens_and_closes(
    e2e_settings, live_server, page, authenticated_user
):
    """Sidebar toggle works."""
    page.goto(f"{live_server.url}/")
    sidebar = page.locator("#aq-sidebar")
    toggle = page.locator("#menu-toggle")

    expect(sidebar).to_be_visible()
    toggle.click()
    expect(sidebar).to_have_attribute("class", re.compile(r"ml-\[-260px\]"))
    toggle.click()
    expect(sidebar).not_to_have_attribute("class", re.compile(r"ml-\[-260px\]"))


def test_user_account_menu_shows_options(
    e2e_settings, live_server, page, authenticated_user
):
    """User account menu opens and shows Manage Account and Logout."""
    page.goto(f"{live_server.url}/")
    page.locator("#account-management-toggle-button").click()
    menu = page.locator("#account-menu-modal")
    expect(menu).to_be_visible()
    expect(menu.get_by_text("Manage Account")).to_be_visible()
    expect(menu.get_by_role("button", name="Logout")).to_be_visible()


def test_navbar_contains_logo_and_text(
    e2e_settings, live_server, page, authenticated_user
):
    """Navbar shows AquiLLM logo and text."""
    page.goto(f"{live_server.url}/")
    expect(page.locator("#nav-logo")).to_be_visible()
    expect(page.locator("nav").get_by_text("AquiLLM")).to_be_visible()


def test_sidebar_has_menu_and_utilities(
    e2e_settings, live_server, page, authenticated_user
):
    """Sidebar includes Menu and Utilities sections."""
    page.goto(f"{live_server.url}/")
    expect(page.locator("#sidebar-header span")).to_have_text("Menu")
    expect(page.locator("#aq-sidebar")).to_contain_text("Utilities")
