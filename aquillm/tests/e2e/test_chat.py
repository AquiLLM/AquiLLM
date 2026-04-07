import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestChatPage:
    """Chat/conversation page behaviour."""

    def test_chat_page_loads(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Chat page loads when navigating to an existing conversation."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/chat/ws_convo/{convo.id}")
        page.wait_for_load_state("networkidle")
        assert f"/chat/ws_convo/{convo.id}" in page.url

    def test_chat_page_has_react_mount(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Chat page contains the React mount point."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/chat/ws_convo/{convo.id}")
        expect(page.locator("#chat-mount")).to_be_visible()

    def test_chat_page_has_correct_title(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Chat page has the expected browser tab title."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/chat/ws_convo/{convo.id}")
        expect(page).to_have_title("AquiLLM -- Chat")

    def test_chat_page_has_sidebar(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Chat page includes sidebar navigation."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/chat/ws_convo/{convo.id}")
        expect(page.locator("#aq-sidebar")).to_be_visible()

    def test_chat_page_has_navbar(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Chat page includes the top navigation bar."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/chat/ws_convo/{convo.id}")
        expect(page.locator("#nav-logo")).to_be_visible()

    def test_new_convo_creates_and_redirects(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Visiting /new_ws_convo/ creates a conversation and redirects to chat."""
        page.goto(f"{live_server.url}/new_ws_convo/")
        page.wait_for_load_state("networkidle")
        assert "/chat/ws_convo/" in page.url
        expect(page.locator("#chat-mount")).to_be_visible()
