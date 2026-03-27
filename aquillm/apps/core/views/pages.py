"""Page views for core app functionality."""
import structlog

from django.apps import apps
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.generic import TemplateView

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import TextChunk
from aquillm.forms import SearchForm
from aquillm.settings import DEBUG

logger = structlog.stdlib.get_logger(__name__)


@require_http_methods(['GET'])
def index(request):
    return render(request, 'aquillm/index.html')


@login_required
@require_http_methods(['GET'])
def react_test(request):
    return render(request, 'aquillm/react_test.html', {"hello_string": "Hello, world!"})


@require_http_methods(['GET', 'POST'])
@login_required
def search(request):
    vector_results = []
    trigram_results = []
    reranked_results = []
    error_message = None

    if request.method == 'POST':
        form = SearchForm(request.user, request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            top_k = form.cleaned_data['top_k']
            collections = form.cleaned_data['collections']
            searchable_docs = Collection.get_user_accessible_documents(request.user, collections=collections)
            vector_results, trigram_results, reranked_results = TextChunk.text_chunk_search(query, top_k, searchable_docs)
        else:
            error_message = "Invalid form submission"
    else:
        form = SearchForm(request.user)

    context = {
        'form': form,
        'reranked_results': reranked_results,
        'vector_results': vector_results,
        'trigram_results': trigram_results,
        'error_message': error_message
    }

    return render(request, 'aquillm/search.html', context)


@require_http_methods(['GET'])
def health_check(request):
    return HttpResponse(status=200)


class UserSettingsPageView(TemplateView):
    template_name = "aquillm/user_settings.html"


if DEBUG:
    @require_http_methods(['GET'])
    @login_required
    def debug_models(request):
        models = apps.get_models()
        model_instances = {model.__name__: list(model.objects.all()) for model in models}
        logger.debug("debug_models loaded %d model classes", len(model_instances))
        return HttpResponse(
            f"debug_models: {len(models)} model classes registered",
            status=200,
            content_type="text/plain",
        )


__all__ = [
    'index',
    'react_test',
    'search',
    'health_check',
    'UserSettingsPageView',
]

if DEBUG:
    __all__.append('debug_models')
