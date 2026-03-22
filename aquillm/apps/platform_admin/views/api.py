"""API views for platform administration."""
import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db.models import Q
from django.http import HttpResponseBadRequest, HttpResponseForbidden, JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from apps.platform_admin.models import EmailWhitelist
from apps.platform_admin.services.feedback_export import parse_query_bounds, stream_feedback_csv_lines

logger = logging.getLogger(__name__)


@login_required
def search_users(request):
    """Search for users by email, username, or name."""
    query = request.GET.get('query', '').strip()
    exclude_current = request.GET.get('exclude_current', 'false').lower() == 'true'
    
    if not query:
        return JsonResponse({'users': []})

    User = get_user_model()
    users = User.objects.filter(
        Q(email__icontains=query) | 
        Q(username__icontains=query) | 
        Q(first_name__icontains=query) | 
        Q(last_name__icontains=query)
    ).distinct()

    if exclude_current:
        users = users.exclude(id=request.user.id)

    user_list = []
    for user in users:
        full_name = f"{user.first_name} {user.last_name}".strip()
        user_list.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': full_name
        })
    
    return JsonResponse({'users': user_list})


@login_required
@require_http_methods(['GET'])
def whitelisted_emails(request):
    """Get all whitelisted email addresses."""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return JsonResponse({'whitelisted': list(EmailWhitelist.objects.all().values_list('email', flat=True))})


@login_required
@require_http_methods(['POST', 'DELETE'])
def whitelisted_email(request, email):
    """Add or remove a whitelisted email address."""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    try:
        validate_email(email)
        if request.method == 'POST':
            EmailWhitelist.objects.get_or_create(email=email)
            return JsonResponse({'status': 'success'})
        if request.method == 'DELETE':
            EmailWhitelist.objects.filter(email=email).delete()
            return JsonResponse({'status': 'success'})
    except ValidationError as e:
        return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@login_required
@require_http_methods(["GET"])
def feedback_ratings_csv(request):
    """Stream CSV of message ratings/feedback (superuser only)."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("Superuser access required")

    start_d, end_d = parse_query_bounds(
        request.GET.get("start_date"),
        request.GET.get("end_date"),
    )

    min_rating = None
    raw_min = request.GET.get("min_rating")
    if raw_min not in (None, ""):
        try:
            min_rating = int(raw_min)
        except ValueError:
            return HttpResponseBadRequest("Invalid min_rating")

    user_number = None
    raw_user = request.GET.get("user_number")
    if raw_user not in (None, ""):
        try:
            user_number = int(raw_user)
        except ValueError:
            return HttpResponseBadRequest("Invalid user_number")

    def content():
        yield from stream_feedback_csv_lines(
            start_date=start_d,
            end_date=end_d,
            min_rating=min_rating,
            user_number=user_number,
        )

    response = StreamingHttpResponse(content(), content_type="text/csv; charset=utf-8")
    fname = f'feedback_ratings_{timezone.now().strftime("%Y%m%d")}.csv'
    response["Content-Disposition"] = f'attachment; filename="{fname}"'
    return response


__all__ = [
    'feedback_ratings_csv',
    'search_users',
    'whitelisted_emails',
    'whitelisted_email',
]
