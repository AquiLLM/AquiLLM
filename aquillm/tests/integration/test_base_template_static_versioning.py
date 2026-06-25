"""Base template should cache-bust the React bundle URL."""
from __future__ import annotations

import re

from django.contrib.auth.models import AnonymousUser
from django.template.loader import render_to_string
from django.test import RequestFactory


def test_base_template_cache_busts_react_bundle_url():
    request = RequestFactory().get("/")
    request.user = AnonymousUser()

    html = render_to_string("aquillm/base.html", request=request)

    assert re.search(r'js/dist/main\.js\?v=\d+', html)
