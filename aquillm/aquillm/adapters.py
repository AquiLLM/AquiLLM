import structlog
from os import getenv

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter

from .models import EmailWhitelist

logger = structlog.stdlib.get_logger(__name__)
class NoDefaultAccounts(DefaultAccountAdapter):
    def is_open_for_signup(self, request):
        return False # No email/password signups allowed

class RestrictDomains(DefaultSocialAccountAdapter):
    def list_apps(self, request, provider=None, client_id=None):
        """
        django-allauth merges SocialApp rows with SOCIALACCOUNT_PROVIDERS from settings.
        The same Google OAuth app in both places yields duplicate provider buttons.
        Keep one entry per (provider id, client_id).
        """
        apps = super().list_apps(request, provider=provider, client_id=client_id)
        seen: set[tuple[str, str]] = set()
        deduped = []
        for app in apps:
            pid = getattr(app, "provider_id", None) or getattr(app, "provider", "") or ""
            cid = (app.client_id or "").strip()
            key = (str(pid), cid)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(app)
        return deduped

    def is_open_for_signup(self, request, sociallogin):
        user = sociallogin.user
        email_domain = user.email.split("@", 1)[1] if user.email and "@" in user.email else ""
        logger.info("obs.auth.oauth_attempt", email=user.email, domain=email_domain)
        allowed_domains = getenv("ALLOWED_EMAIL_DOMAINS", default="").split(",")
        allowed_emails = getenv("ALLOWED_EMAIL_ADDRESSES", default="").split(",")
        allow = (user.email.split('@')[1] in allowed_domains or
                user.email in allowed_emails or
                user.email in EmailWhitelist.objects.values_list('email', flat=True))
        logger.info("obs.auth.oauth_decision", allow=allow)
        return allow
