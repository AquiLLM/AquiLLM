import re

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]

# The sidebar also renders a conversation list partial with the same IDs,
# so we scope selectors to the main content area to avoid strict-mode violations.
CONTENT = "#page-content-wrapper"


class TestConversationsList:
    """Conversations list page behaviour."""

    def test_conversations_page_renders(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Conversations page loads with correct title."""
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        expect(page).to_have_title("Your Conversations")
        expect(page.locator(f"{CONTENT} h1")).to_contain_text("Your Conversations")

    def test_empty_conversations_shows_message(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """When no conversations exist, a helpful message is shown."""
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        expect(page.locator(f"{CONTENT} .no-conversations")).to_contain_text(
            "You don't have any conversations yet"
        )

    def test_conversations_list_shows_conversations(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Created conversations appear in the list."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="Test Chat"
        )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        row = page.locator(f"{CONTENT} #convo-{convo.id}")
        expect(row).to_be_visible()
        expect(row).to_contain_text("Test Chat")

    def test_untitled_conversation_shows_placeholder(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Conversations without a name show 'Untitled Conversation'."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_user)
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        row = page.locator(f"{CONTENT} #convo-{convo.id}")
        expect(row).to_contain_text("Untitled Conversation")

    def test_conversation_link_navigates_to_chat(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking a conversation link navigates to the chat page."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="Navigate Test"
        )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        page.locator(f"{CONTENT} #convo-{convo.id} a").click()
        page.wait_for_load_state("networkidle")
        assert f"/chat/ws_convo/{convo.id}" in page.url

    def test_conversations_ordered_by_newest_first(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Conversations are listed with the newest first."""
        from apps.chat.models import WSConversation

        WSConversation.objects.create(owner=authenticated_user, name="Older Chat")
        WSConversation.objects.create(owner=authenticated_user, name="Newer Chat")
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        rows = page.locator(f"{CONTENT} tbody tr")
        first_text = rows.nth(0).inner_text()
        assert "Newer Chat" in first_text

    def test_multiple_conversations_all_visible(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Multiple conversations are all rendered in the table."""
        from apps.chat.models import WSConversation

        for i in range(5):
            WSConversation.objects.create(
                owner=authenticated_user, name=f"Convo {i}"
            )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        rows = page.locator(f"{CONTENT} tbody tr")
        expect(rows).to_have_count(5)


class TestConversationDelete:
    """Conversation deletion flow."""

    def test_delete_button_shows_confirmation(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking Delete shows the confirmation prompt."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="To Delete"
        )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        page.locator(f"{CONTENT} #delete-button-{convo.id}").click()
        confirm_div = page.locator(f"{CONTENT} #yes-no-{convo.id}")
        expect(confirm_div).to_be_visible()
        expect(confirm_div).to_contain_text("Yes, Really")
        expect(confirm_div).to_contain_text("No, Not Really")

    def test_cancel_delete_hides_confirmation(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Clicking 'No, Not Really' hides the confirmation and restores the Delete button."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="Keep This"
        )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        page.locator(f"{CONTENT} #delete-button-{convo.id}").click()
        # Use the scoped locator to avoid sidebar duplicates
        page.locator(f"{CONTENT} #yes-no-{convo.id}").get_by_text(
            "No, Not Really"
        ).click()
        confirm_div = page.locator(f"{CONTENT} #yes-no-{convo.id}")
        expect(confirm_div).not_to_be_visible()
        expect(page.locator(f"{CONTENT} #delete-button-{convo.id}")).to_be_visible()

    def test_confirm_delete_removes_conversation_from_db(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Confirming deletion deletes the conversation from the database."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="Delete Me"
        )
        convo_id = convo.id
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")
        page.locator(f"{CONTENT} #delete-button-{convo_id}").click()
        page.locator(
            f"{CONTENT} #yes-no-{convo_id} button[type='submit']"
        ).click()
        # Wait for the fetch DELETE to complete
        page.wait_for_timeout(1000)
        assert not WSConversation.objects.filter(pk=convo_id).exists()


class TestNewConversation:
    """Creating new conversations."""

    def test_new_conversation_redirects_to_chat(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Visiting /new_ws_convo/ creates a conversation and redirects to it."""
        page.goto(f"{live_server.url}/new_ws_convo/")
        page.wait_for_load_state("networkidle")
        assert "/chat/ws_convo/" in page.url
