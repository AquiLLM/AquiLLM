"""Views for platform_admin app."""
from .api import (
    search_users,
    whitelisted_emails,
    whitelisted_email,
)
from .pages import (
    gemini_cost_monitor,
    email_whitelist,
)

__all__ = [
    # API views
    'search_users',
    'whitelisted_emails',
    'whitelisted_email',
    # Page views
    'gemini_cost_monitor',
    'email_whitelist',
]
