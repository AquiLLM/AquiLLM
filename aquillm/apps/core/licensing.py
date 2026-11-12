"""Public access to license documents and an operator's corresponding source."""

from pathlib import Path
from urllib.parse import urlsplit

from django.conf import settings
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_safe

REPO_ROOT = Path(__file__).resolve().parents[3]


def deployment_source_url():
    """Never turn a missing source offer into an incorrect upstream-source claim."""
    value = getattr(settings, "AQUILLM_SOURCE_URL", "").strip()
    try:
        parsed = urlsplit(value)
        # urlsplit defers invalid/non-numeric port errors until this property is read.
        parsed.port
        valid = (
            parsed.scheme in {"https", "http"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not any(ord(char) < 33 for char in value)
        )
    except ValueError:
        valid = False
    return value if valid else ""


@require_safe
def licensing_page(request):
    # This public page needs neither an authenticated user nor database-backed
    # navigation/context processors, including during an application outage.
    html = render_to_string("aquillm/licensing.html", {
        "deployment_source_url": deployment_source_url(),
    })
    return HttpResponse(html)


@require_safe
def project_license(request):
    return HttpResponse((REPO_ROOT / "LICENSE").read_text(encoding="utf-8"),
                        content_type="text/plain; charset=utf-8")


@require_safe
def third_party_notices(request):
    return HttpResponse((REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"),
                        content_type="text/plain; charset=utf-8")
