"""Page views for chat functionality."""
import structlog
import os

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from apps.chat.models import WSConversation

logger = structlog.stdlib.get_logger(__name__)


@require_http_methods(["GET"])
@login_required
def new_ws_convo(request):
    convo = WSConversation(owner=request.user)
    convo.save()
    return redirect("ws_convo", convo_id=convo.id)


@require_http_methods(['GET'])
@login_required
def ws_convo(request, convo_id):
    """Display a WebSocket conversation page."""
    context_limit_raw = (
        (os.getenv("OPENAI_CONTEXT_LIMIT", "") or "").strip()
        or (os.getenv("VLLM_MAX_MODEL_LEN", "") or "").strip()
    )
    try:
        context_limit = int(context_limit_raw)
    except (TypeError, ValueError):
        context_limit = 0
    if context_limit <= 0:
        context_limit = 200000

    return render(
        request,
        'aquillm/ws_convo.html',
        {
            'convo_id': convo_id,
            'context_limit': context_limit,
        },
    )


@require_http_methods(['DELETE'])
@login_required
def delete_ws_convo(request, convo_id):
    """Delete a WebSocket conversation."""
    convo = get_object_or_404(WSConversation, pk=convo_id)
    if convo.owner != request.user:
        return HttpResponseForbidden("User does not have permission to delete this conversation.")
    convo.delete()
    return HttpResponse(status=200)


@require_http_methods(['GET'])
@login_required
def user_ws_convos(request):
    """List all conversations for the current user."""
    convos = WSConversation.objects.filter(owner=request.user).order_by('-created_at')
    return render(request, 'aquillm/user_ws_convos.html', {'conversations': convos})


__all__ = [
    "new_ws_convo",
    "ws_convo",
    "delete_ws_convo",
    "user_ws_convos",
]
