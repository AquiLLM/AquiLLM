"""E2E tests for error states and edge-case UI behavior.

These tests verify the app handles unexpected inputs, missing data,
and boundary conditions gracefully — the kind of scenarios that
produce real user-facing bugs.
"""

import pytest
from django.contrib.auth import get_user_model
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]

User = get_user_model()


class TestInvalidConversationAccess:
    """Accessing conversations that don't exist or belong to someone else."""

    def test_nonexistent_conversation_shows_error(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Visiting a conversation ID that doesn't exist shows an error
        rather than a blank or broken page."""
        page.goto(f"{channels_live_server.url}/chat/ws_convo/999999")

        # The WebSocket consumer sends {"exception": "Invalid chat_id"}.
        # The React Chat component should display this error.
        error_banner = page.locator("text=Invalid chat_id")
        expect(error_banner).to_be_visible(timeout=10000)

    def test_other_users_conversation_shows_error(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Accessing another user's conversation shows an error."""
        from apps.chat.models import WSConversation

        other_user = User.objects.create_user(
            username="other_user", email="other@example.com", password="pass123"
        )
        convo = WSConversation.objects.create(owner=other_user, name="Private Chat")

        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        error_banner = page.locator("text=Invalid chat_id")
        expect(error_banner).to_be_visible(timeout=10000)

    def test_other_users_conversation_blocks_messaging(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Cannot send messages in another user's conversation."""
        from apps.chat.models import Message, WSConversation

        other_user = User.objects.create_user(
            username="other_user2", email="other2@example.com", password="pass123"
        )
        convo = WSConversation.objects.create(owner=other_user)

        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")
        # Wait for the error to appear
        expect(page.locator("text=Invalid chat_id")).to_be_visible(timeout=10000)

        # Even if we try to type and send, no messages should be persisted
        textarea = page.locator("#message-input")
        if textarea.is_visible():
            textarea.fill("Sneaky message")
            textarea.press("Enter")
            page.wait_for_timeout(2000)

        assert Message.objects.filter(conversation=convo).count() == 0


class TestLongContent:
    """Messages and data with unusual lengths."""

    def test_long_message_wraps_correctly(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """A very long message without spaces should wrap inside the chat
        bubble rather than overflowing the container."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        long_text = "A" * 2000
        textarea.fill(long_text)
        textarea.press("Enter")

        user_msg = page.locator(".user-message").first
        expect(user_msg).to_be_visible(timeout=10000)

        # The message bubble should not be wider than the chat container
        msg_box = user_msg.bounding_box()
        container_box = page.locator("#chat-mount").bounding_box()
        assert msg_box is not None and container_box is not None
        assert msg_box["width"] <= container_box["width"], (
            f"Message bubble ({msg_box['width']}px) overflows container ({container_box['width']}px)"
        )

    @pytest.mark.xfail(reason="Frontend bug: long convo name overflows table row (6279px on 1280px viewport)")
    def test_long_conversation_name_does_not_break_list(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """A very long conversation name doesn't break the conversations list layout."""
        from apps.chat.models import WSConversation

        long_name = "X" * 500
        convo = WSConversation.objects.create(
            owner=authenticated_user, name=long_name
        )
        page.goto(f"{live_server.url}/aquillm/user_ws_convos/")

        CONTENT = "#page-content-wrapper"
        row = page.locator(f"{CONTENT} #convo-{convo.id}")
        expect(row).to_be_visible()

        # Row should not be wider than the page
        row_box = row.bounding_box()
        viewport = page.viewport_size
        assert row_box is not None and viewport is not None
        assert row_box["width"] <= viewport["width"]


class TestSpecialCharacters:
    """Messages containing HTML, scripts, or unusual characters."""

    def test_html_in_message_is_not_rendered_as_html(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """HTML tags in a user message should be escaped, not rendered."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        textarea.fill('<img src=x onerror="alert(1)">')
        textarea.press("Enter")

        user_msg = page.locator(".user-message").first
        expect(user_msg).to_be_visible(timeout=10000)

        # The raw HTML should appear as text, not as a rendered element
        expect(user_msg).to_contain_text("<img")
        # No actual <img> element should exist inside the message bubble
        assert user_msg.locator("img").count() == 0

    def test_script_tag_in_message_is_not_executed(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Script tags in messages must not execute (XSS prevention)."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        # Set a sentinel on window that the script would overwrite
        page.evaluate("window.__xss_test = false")
        textarea.fill('<script>window.__xss_test = true</script>')
        textarea.press("Enter")

        user_msg = page.locator(".user-message").first
        expect(user_msg).to_be_visible(timeout=10000)

        # Sentinel must remain false — script was not executed
        assert page.evaluate("window.__xss_test") is False

    def test_unicode_emoji_renders_correctly(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Unicode emoji and multi-byte characters render without corruption."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        emoji_text = "Hello 🌍🔬📚 — em dash — «guillemets» — 日本語テスト"
        textarea.fill(emoji_text)
        textarea.press("Enter")

        user_msg = page.locator(".user-message").first
        expect(user_msg).to_be_visible(timeout=10000)
        expect(user_msg).to_contain_text("🌍🔬📚")
        expect(user_msg).to_contain_text("日本語テスト")


class TestEmptyAndWhitespace:
    """Edge cases around empty or whitespace-only inputs."""

    def test_empty_message_not_sent(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Clicking send with an empty input should not create a message."""
        from apps.chat.models import Message, WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        # Click send without typing anything
        page.locator('button[title="Send Message"]').click()
        page.wait_for_timeout(2000)

        assert page.locator(".user-message").count() == 0
        assert Message.objects.filter(conversation=convo).count() == 0

    def test_whitespace_only_message_not_sent(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Sending only whitespace should not create a message."""
        from apps.chat.models import Message, WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_enabled(timeout=10000)

        textarea.fill("   \n\n   ")
        textarea.press("Enter")
        page.wait_for_timeout(2000)

        assert page.locator(".user-message").count() == 0
        assert Message.objects.filter(conversation=convo).count() == 0


class TestDeletedConversationNavigation:
    """Navigating to or interacting with a conversation after it's been deleted."""

    def test_deleted_conversation_shows_error(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Navigating to a deleted conversation's URL shows an error."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        convo_id = convo.id
        convo.delete()

        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo_id}")
        error_banner = page.locator("text=Invalid chat_id")
        expect(error_banner).to_be_visible(timeout=10000)
