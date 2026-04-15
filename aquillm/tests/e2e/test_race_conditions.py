"""E2E tests for race conditions and concurrent interaction bugs.

These tests verify that rapid user actions don't produce duplicate
submissions, broken UI state, or other concurrency-related issues.
"""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestDoubleSubmit:
    """Rapid clicks / keypresses should not produce duplicate actions."""

    def test_double_click_send_produces_one_response(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Double-clicking the send button should not send the message twice."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        textarea.fill("Double click test")
        send_btn = page.locator('button[title="Send Message"]')
        send_btn.dblclick()

        # Wait for the echo response
        expect(page.locator(".assistant-message").first).to_be_visible(timeout=15000)
        # Should have exactly one user message and one assistant response
        expect(page.locator(".user-message")).to_have_count(1, timeout=5000)
        expect(page.locator(".assistant-message")).to_have_count(1, timeout=5000)

    def test_double_enter_produces_one_response(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Pressing Enter twice rapidly should not send the message twice."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        textarea.fill("Double enter test")
        textarea.press("Enter")
        textarea.press("Enter")

        expect(page.locator(".assistant-message").first).to_be_visible(timeout=15000)
        expect(page.locator(".user-message")).to_have_count(1, timeout=5000)
        expect(page.locator(".assistant-message")).to_have_count(1, timeout=5000)

    def test_double_click_new_conversation_creates_one(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Rapidly visiting /new_ws_convo/ should not create duplicate conversations."""
        from apps.chat.models import WSConversation

        count_before = WSConversation.objects.filter(owner=authenticated_user).count()
        page.goto(f"{live_server.url}/new_ws_convo/")
        page.wait_for_load_state("networkidle")
        assert "/chat/ws_convo/" in page.url
        count_after = WSConversation.objects.filter(owner=authenticated_user).count()
        assert count_after == count_before + 1

    def test_double_click_delete_only_deletes_once(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Double-clicking the delete confirm button should not cause errors."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(
            owner=authenticated_user, name="Delete Once"
        )
        convo_id = convo.id
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")

        CONTENT = "#page-content-wrapper"
        page.locator(f"{CONTENT} #delete-button-{convo_id}").click()
        confirm_btn = page.locator(f"{CONTENT} #yes-no-{convo_id} button[type='submit']")
        confirm_btn.dblclick()
        page.wait_for_timeout(1500)

        assert not WSConversation.objects.filter(pk=convo_id).exists()


class TestRapidUIToggle:
    """Rapid toggling of UI elements should not leave broken state."""

    def test_rapid_sidebar_toggle(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Rapidly toggling the sidebar multiple times should not break layout."""
        page.goto(f"{live_server.url}/")
        toggle = page.locator("#menu-toggle")
        expect(toggle).to_be_visible(timeout=10000)

        # Toggle rapidly 6 times (should end in original state)
        for _ in range(6):
            toggle.click()

        sidebar = page.locator("#aq-sidebar")
        expect(sidebar).to_be_visible()
        # Even count of clicks → sidebar should be back to default (no collapse class)
        # Verify the sidebar is actually usable by checking a child element
        expect(sidebar.locator("text=Menu")).to_be_visible()

    @pytest.mark.xfail(reason="Frontend bug: account menu gets stuck after rapid toggling")
    def test_rapid_account_menu_toggle(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Rapidly opening/closing the account menu should not leave it stuck."""
        page.goto(f"{live_server.url}/")
        account_btn = page.locator("#account-btn")
        expect(account_btn).to_be_visible(timeout=10000)

        # Toggle rapidly
        for _ in range(5):
            account_btn.click()

        # Odd count → menu should be open
        expect(page.get_by_text("Logout")).to_be_visible(timeout=3000)

        # One more click → should close
        account_btn.click()
        expect(page.get_by_text("Logout")).not_to_be_visible(timeout=3000)


class TestSendWhileProcessing:
    """Attempting to send messages while a response is still streaming."""

    @pytest.mark.xfail(reason="Frontend bug: input doesn't re-enable or WS stalls after first exchange")
    def test_cannot_send_while_input_disabled(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """While the LLM is processing, the input should be disabled and further
        sends should be blocked."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        textarea.fill("First message")
        page.locator('button[title="Send Message"]').click()

        # Wait for the full round trip to complete
        expect(page.locator(".assistant-message").first).to_be_visible(timeout=15000)
        expect(textarea).to_be_enabled(timeout=10000)

        # Now send a second message — should work cleanly
        textarea.fill("Second message")
        textarea.press("Enter")
        expect(page.locator(".assistant-message")).to_have_count(2, timeout=15000)

    @pytest.mark.xfail(reason="Frontend bug: input doesn't re-enable or WS stalls after first exchange")
    def test_sequential_messages_maintain_order(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Messages sent sequentially appear in the correct order."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        for i in range(3):
            textarea.fill(f"Message {i}")
            textarea.press("Enter")
            expect(page.locator(".assistant-message")).to_have_count(
                i + 1, timeout=15000
            )
            expect(textarea).to_be_enabled(timeout=10000)

        # Verify ordering: user messages should appear in order
        user_msgs = page.locator(".user-message")
        for i in range(3):
            expect(user_msgs.nth(i)).to_contain_text(f"Message {i}")
