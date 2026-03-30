"""Page views for bug reports."""
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render
from django.views.decorators.http import require_http_methods


@require_http_methods(["GET"])
@login_required
@user_passes_test(lambda u: u.is_staff)
def bug_reports_admin(request):
    """Render the bug reports admin dashboard."""
    return render(request, "aquillm/bug_reports_admin.html")
