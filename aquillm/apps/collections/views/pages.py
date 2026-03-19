"""Page views for collection management."""
import logging

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection, CollectionPermission
from aquillm.forms import NewCollectionForm

logger = logging.getLogger(__name__)


@require_http_methods(['GET', 'POST'])
@login_required
def user_collections(request):
    """View and create user collections."""
    if request.method == 'POST':
        form = NewCollectionForm(request.POST, user=request.user)
        if form.is_valid():
            data = form.cleaned_data
            name = data['name']
            viewers = data['viewers']
            editors = data['editors']
            admins = data['admins']
            with transaction.atomic():
                col = Collection.objects.create(name=name)
                CollectionPermission.objects.create(user=request.user, collection=col, permission='MANAGE')
                for user in admins:
                    CollectionPermission.objects.create(user=user, collection=col, permission='MANAGE')
                for user in editors:
                    CollectionPermission.objects.create(user=user, collection=col, permission='EDIT')
                for user in viewers:
                    CollectionPermission.objects.create(user=user, collection=col, permission='VIEW')
        else:
            colperms = CollectionPermission.objects.filter(user=request.user)
            status_message = "Invalid Form Input"
            return render(request, "aquillm/user_collections.html", {'col_perms': colperms, 'form': form, 'status_message': status_message})

        return redirect('collection', col_id=col.pk)
    else:
        colperms = CollectionPermission.objects.filter(user=request.user)
        form = NewCollectionForm(user=request.user)
        return render(request, "aquillm/user_collections.html", {'col_perms': colperms, 'form': form})


@require_http_methods(['GET'])
@login_required
def collection(request, col_id):
    """View to display a collection and its contents."""
    try:
        collection_obj = get_object_or_404(Collection, pk=col_id)
        if not collection_obj.user_can_view(request.user):
            return HttpResponseForbidden("User does not have permission to view this collection.")
        
        available_collections = Collection.objects.filter_by_user_perm(request.user, 'EDIT')
        
        return render(request, 'aquillm/collection.html', {
            'collection': collection_obj,
            'path': collection_obj.get_path(),
            'can_edit': collection_obj.user_can_edit(request.user),
            'can_delete': collection_obj.user_can_manage(request.user),
            'available_collections': available_collections,
        })
    except Exception as e:
        logger.error(f"Error in collection view: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


__all__ = [
    'user_collections',
    'collection',
]
