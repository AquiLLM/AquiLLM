from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from os import getenv
from .models import EmailWhitelist
class NoDefaultAccounts(DefaultAccountAdapter):
    def is_open_for_signup(self, request):
        return False # No email/password signups allowed

class RestrictDomains(DefaultSocialAccountAdapter):
    def is_open_for_signup(self, request, sociallogin):
        user = sociallogin.user
        print(f"ATTEMPTED SIGNUP: {user.email}")
        allowed_domains = getenv("ALLOWED_EMAIL_DOMAINS", default="").split(",")
        allowed_emails = getenv("ALLOWED_EMAIL_ADDRESSES", default="").split(",")
        allow = (user.email.split('@')[1] in allowed_domains or
                user.email in allowed_emails or
                user.email in EmailWhitelist.objects.values_list('email', flat=True))
        print(f"ALLOWED={allow}")
        return allow
