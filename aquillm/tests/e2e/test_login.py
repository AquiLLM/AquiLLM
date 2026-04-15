import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


def test_login_page_renders(e2e_settings, live_server, page):
    """Login page loads and shows the login form."""
    page.goto(f"{live_server.url}/accounts/login/")
    expect(page.locator('input[name="login"]')).to_be_visible()
    expect(page.locator('input[name="password"]')).to_be_visible()
    expect(page.locator('button[type="submit"]')).to_be_visible()


def test_login_form_rejects_invalid_credentials(e2e_settings, live_server, page):
    """Submitting wrong credentials shows an error, not a crash."""
    page.goto(f"{live_server.url}/accounts/login/")
    page.locator('input[name="login"]').fill("nonexistent")
    page.locator('input[name="password"]').fill("wrongpassword")
    page.locator('button[type="submit"]').click()
    page.wait_for_load_state("networkidle")
    # allauth shows error messages on failed login
    expect(page.locator(".alert, .errorlist, [role='alert']").first).to_be_visible()


def test_login_with_valid_credentials_redirects_to_home(
    e2e_settings, db, live_server, page
):
    """Logging in with valid credentials redirects to /."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    User.objects.create_user(
        username="logintest", email="login@test.com", password="testpass123"
    )
    page.goto(f"{live_server.url}/accounts/login/")
    page.locator('input[name="login"]').fill("logintest")
    page.locator('input[name="password"]').fill("testpass123")
    page.locator('button[type="submit"]').click()
    page.wait_for_url(f"{live_server.url}/")
    # After login, we should be on the homepage
    assert page.url == f"{live_server.url}/"


def test_login_page_layout_is_centered(e2e_settings, live_server, page):
    """Login card is horizontally centered within its containing section."""
    page.goto(f"{live_server.url}/accounts/login/")
    card = page.locator("#content")
    # The login card lives inside a <section> that is the right column of a
    # two-column layout at wide viewports.  Check centering within the section,
    # not the full viewport.
    section = page.locator("section").filter(has=card)
    card_box = card.bounding_box()
    section_box = section.bounding_box()
    assert card_box is not None
    assert section_box is not None
    card_center = card_box["x"] + card_box["width"] / 2
    section_center = section_box["x"] + section_box["width"] / 2
    assert abs(card_center - section_center) <= 20
