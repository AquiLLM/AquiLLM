"""Short-token storage for feedback dashboard queries.

The LLM-generated dashboard links use a short opaque token instead of a
base64-encoded query string. Background: gpt-4o (and other LLMs) sometimes
drop characters when transcribing long opaque strings into their replies.
A 7-character token has too few characters to drop without producing an
obvious typo, and it's deterministic to copy.

Manually-shared dashboard links (the dashboard's "Copy link" button) keep
using the base64-in-URL form — those URLs are self-contained and never
expire, which matters for links pasted into Slack or tickets.
"""
from __future__ import annotations

import secrets

from django.core.cache import cache


# 90 days. Long enough that a teammate can come back to a chat-shared link
# weeks later; short enough that we don't grow Redis indefinitely.
_TTL_SECONDS = 90 * 24 * 60 * 60

_KEY_PREFIX = "feedback_q:"


def mint_token(query: str) -> str:
    """Store a query under a fresh short token; return the token."""
    # 6 random bytes → 8 url-safe base64 chars. ~2.8e14 combinations.
    token = secrets.token_urlsafe(6)
    cache.set(f"{_KEY_PREFIX}{token}", query, _TTL_SECONDS)
    return token


def resolve_token(token: str) -> str | None:
    """Return the query for a token, or None if missing/expired."""
    return cache.get(f"{_KEY_PREFIX}{token}")
