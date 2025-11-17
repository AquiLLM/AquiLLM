from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseForbidden
from aquillm.models import WSConversation 

from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from django.urls import path


@require_http_methods(['GET'])
@login_required
def new_ws_convo(request):
    convo = WSConversation(owner=request.user)
    convo.save()
    return redirect('ws_convo', convo_id=convo.id) # type: ignore

@require_http_methods(['GET'])
@login_required
def ws_convo(request, convo_id):
    return render(request, 'aquillm/ws_convo.html', {'convo_id': convo_id})

@require_http_methods(['DELETE'])
@login_required
def delete_ws_convo(request, convo_id):
    convo = get_object_or_404(WSConversation, pk=convo_id)
    if convo.owner != request.user:
        return HttpResponseForbidden("User does not have permission to delete this conversation.")
    convo.delete()
    return HttpResponse(status=200)    

urlpatterns = [
    path("ws_convo/<int:convo_id>", ws_convo, name="ws_convo"),
    path("delete_ws_convo/<int:convo_id>", delete_ws_convo, name="delete_ws_convo"),
    path("new_ws_convo/", new_ws_convo, name="new_ws_convo"),
]