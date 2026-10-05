"""API views for chat functionality."""
import json

import structlog

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.chat.models import ConversationFile, WSConversation

logger = structlog.stdlib.get_logger(__name__)


@login_required
@require_http_methods(["POST"])
def rename_ws_convo(request, convo_id):
    """Rename owned chat metadata without changing transcript/activity identity."""
    convo = get_object_or_404(WSConversation, pk=convo_id, owner=request.user)
    try:
        payload = json.loads(request.body)
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid request body."}, status=400)
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
        return JsonResponse(
            {"error": "Enter a chat name between 1 and 200 characters."}, status=400
        )
    name = name.strip()
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return JsonResponse({"error": "Use a single line for the chat name."}, status=400)
    changed = WSConversation.objects.filter(pk=convo.pk, owner=request.user).update(
        name=name, name_is_manual=True
    )
    if not changed:
        return JsonResponse({"error": "This chat is no longer available."}, status=404)
    return JsonResponse({"id": convo.pk, "name": name})


@login_required
@require_http_methods(['GET'])
def conversation_file(request, convo_file_id):
    """Download a file attached to a conversation."""
    convo_file = get_object_or_404(ConversationFile, pk=convo_file_id)
    if not convo_file.conversation.owner == request.user:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return FileResponse(convo_file.file, as_attachment=True)


__all__ = [
    'conversation_file',
    'rename_ws_convo',
]
