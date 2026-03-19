"""API views for chat functionality."""
import logging

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.chat.models import ConversationFile

logger = logging.getLogger(__name__)


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
]
