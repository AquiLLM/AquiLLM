import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestSearchPageLoads:
    """Search page renders correctly for authenticated users."""

    def test_search_page_renders_form(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Search page shows the query form with all expected fields."""
        page.goto(f"{live_server.url}/aquillm/search/")
        expect(page.locator('textarea[name="query"]')).to_be_visible()
        expect(page.locator('input[name="top_k"]')).to_be_visible()
        expect(page.locator('input[type="submit"][value="Submit"]')).to_be_visible()

    def test_search_page_has_correct_title(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Search page has the expected browser tab title."""
        page.goto(f"{live_server.url}/aquillm/search/")
        expect(page).to_have_title("AquiLLM Search")

    def test_search_top_k_default_value(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Top-k input defaults to 5."""
        page.goto(f"{live_server.url}/aquillm/search/")
        top_k = page.locator('input[name="top_k"]')
        expect(top_k).to_have_value("5")

    def test_search_query_placeholder(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Query textarea has the expected placeholder text."""
        page.goto(f"{live_server.url}/aquillm/search/")
        textarea = page.locator('textarea[name="query"]')
        expect(textarea).to_have_attribute("placeholder", "Send a message")

    def test_search_form_posts_to_correct_url(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Search form action points to /search/."""
        page.goto(f"{live_server.url}/aquillm/search/")
        form = page.locator("form[action='/search/']")
        expect(form).to_be_visible()

    def test_search_empty_submission_stays_on_page(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Submitting an empty search form stays on the search page (HTML validation)."""
        page.goto(f"{live_server.url}/aquillm/search/")
        # Clear query and submit
        page.locator('textarea[name="query"]').fill("")
        page.locator('input[type="submit"]').click()
        page.wait_for_load_state("networkidle")
        # Should remain on search page (form validation prevents empty submit or
        # server rejects and re-renders the form)
        assert "/search/" in page.url or "/aquillm/search/" in page.url

    def test_search_top_k_respects_min_max(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Top-k field has min=1 and max=200 attributes."""
        page.goto(f"{live_server.url}/aquillm/search/")
        top_k = page.locator('input[name="top_k"]')
        expect(top_k).to_have_attribute("min", "1")
        expect(top_k).to_have_attribute("max", "200")
