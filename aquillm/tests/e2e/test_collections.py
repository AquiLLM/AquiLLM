import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestCollectionsPage:
    """Collections page behaviour."""

    def test_collections_page_loads(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Collections page loads for authenticated users."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        page.wait_for_load_state("networkidle")
        assert "/accounts/login/" not in page.url

    def test_collections_page_has_react_mount(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Collections page contains the React mount point."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        expect(page.locator("#collections-page")).to_be_visible()

    def test_collections_page_has_correct_title(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Collections page has the expected browser tab title."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        expect(page).to_have_title("AquiLLM -- Collections")

    def test_collections_page_has_sidebar(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Collections page includes sidebar navigation."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        expect(page.locator("#aq-sidebar")).to_be_visible()
