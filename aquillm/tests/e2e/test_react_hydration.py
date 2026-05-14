"""E2E tests verifying React components actually render content.

Existing tests only check that the mount-point <div> exists in the DOM.
These tests verify that the JS bundle loads and React renders meaningful
UI inside each mount point — catching broken builds, missing static
files, and hydration failures.
"""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]


class TestCollectionsReactRenders:
    """Verify the Collections page React app boots and renders real content."""

    def test_collections_renders_heading(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """React renders the 'My Collections' heading inside the mount point."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        heading = page.locator("#collections-page h1")
        expect(heading).to_contain_text("My Collections", timeout=10000)

    def test_collections_renders_new_collection_button(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """The 'New Collection' action button is rendered by React."""
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        btn = page.get_by_role("button", name="New Collection")
        expect(btn).to_be_visible(timeout=10000)

    def test_collections_no_console_errors(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """No JS console errors when the collections page loads."""
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"{live_server.url}/aquillm/user_collections/")
        # Wait for React to mount
        expect(page.locator("#collections-page h1")).to_be_visible(timeout=10000)
        assert errors == [], f"Console errors on collections page: {errors}"


class TestUserSettingsReactRenders:
    """Verify the User Settings page React app boots and renders real content."""

    def test_settings_renders_heading(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """React renders the 'User Settings' heading."""
        page.goto(f"{live_server.url}/user-settings/")
        heading = page.get_by_role("heading", name="User Settings")
        expect(heading).to_be_visible(timeout=10000)

    def test_settings_renders_theme_selector(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Theme color scheme dropdown is rendered by React."""
        page.goto(f"{live_server.url}/user-settings/")
        select = page.locator("#color_scheme")
        expect(select).to_be_visible(timeout=10000)

    def test_settings_renders_font_selector(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """Font family dropdown is rendered by React."""
        page.goto(f"{live_server.url}/user-settings/")
        select = page.locator("#font_family")
        expect(select).to_be_visible(timeout=10000)

    def test_settings_renders_save_button(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """The 'Save Settings' button is rendered by React."""
        page.goto(f"{live_server.url}/user-settings/")
        btn = page.locator('[data-testid="save-theme-settings"]')
        expect(btn).to_be_visible(timeout=10000)

    def test_settings_renders_zotero_section(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """The Zotero Integration section is rendered by React."""
        page.goto(f"{live_server.url}/user-settings/")
        expect(page.get_by_text("Zotero Integration")).to_be_visible(timeout=10000)

    def test_settings_no_console_errors(
        self, e2e_settings, live_server, page, authenticated_user
    ):
        """No JS console errors when the settings page loads."""
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"{live_server.url}/user-settings/")
        expect(page.locator("#color_scheme")).to_be_visible(timeout=10000)
        assert errors == [], f"Console errors on settings page: {errors}"


class TestChatReactRenders:
    """Verify the Chat page React app boots and renders real content."""

    def test_chat_renders_input_dock(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """React renders the chat input textarea and send button."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        textarea = page.locator("#message-input")
        expect(textarea).to_be_visible(timeout=10000)

        send_btn = page.locator('button[title="Send Message"]')
        expect(send_btn).to_be_visible()

    def test_chat_renders_message_container(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """React renders the scrollable message container."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")

        # The top-level flex container rendered by React (direct child of mount)
        container = page.locator("#chat-mount > div.flex.flex-col")
        expect(container).to_be_visible(timeout=10000)

    def test_chat_no_console_errors(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """No JS console errors when the chat page loads."""
        from apps.chat.models import WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")
        expect(page.locator("#message-input")).to_be_visible(timeout=10000)
        assert errors == [], f"Console errors on chat page: {errors}"

    def test_chat_loads_existing_messages(
        self, e2e_settings, channels_live_server, page, authenticated_ws_user, echo_llm
    ):
        """Chat page renders messages that were previously persisted."""
        from apps.chat.models import Message, WSConversation

        convo = WSConversation.objects.create(owner=authenticated_ws_user)
        Message.objects.create(
            conversation=convo, role="user", content="Previously sent message",
            sequence_number=1,
        )
        Message.objects.create(
            conversation=convo, role="assistant", content="Previous reply",
            sequence_number=2,
        )

        page.goto(f"{channels_live_server.url}/chat/ws_convo/{convo.id}")
        expect(page.locator(".user-message")).to_contain_text(
            "Previously sent message", timeout=15000
        )
        expect(page.locator(".assistant-message")).to_contain_text("Previous reply")
