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
    def is_open_for_signup(self, request, sociallogin):
        user = sociallogin.user
        email_domain = user.email.split("@", 1)[1] if user.email and "@" in user.email else ""
        logger.info(
            "OAuth signup attempt email=%s domain=%s",
            user.email,
            email_domain,
        )
        allowed_domains = getenv("ALLOWED_EMAIL_DOMAINS", default="").split(",")
        allowed_emails = getenv("ALLOWED_EMAIL_ADDRESSES", default="").split(",")
        allow = (user.email.split('@')[1] in allowed_domains or
                user.email in allowed_emails or
                user.email in EmailWhitelist.objects.values_list('email', flat=True))
        logger.info("OAuth signup decision allow=%s", allow)
        return allow
