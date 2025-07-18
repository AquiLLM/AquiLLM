from django.shortcuts import render, redirect
from django.apps import apps

from django.core.files.base import ContentFile
import logging
import re
import io
import gzip
import tarfile
from django.urls import path
from xml.dom import minidom
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.http import HttpResponse, Http404, HttpResponseForbidden
from django.shortcuts import get_object_or_404
from pgvector.django import L2Distance
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.exceptions import ObjectDoesNotExist


from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import requires_csrf_token

from .forms import SearchForm, ArXiVForm, PDFDocumentForm, VTTDocumentForm, NewCollectionForm, HandwrittenNotesForm
from .models import TextChunk, TeXDocument, PDFDocument, VTTDocument, Collection, CollectionPermission, WSConversation, DESCENDED_FROM_DOCUMENT, HandwrittenNotesDocument
from . import vtt
from .settings import DEBUG

import requests
from django.http import JsonResponse
from django.forms.models import model_to_dict
import json
import anthropic
import os
import base64
import hashlib
import uuid

from .forms import HandwrittenNotesForm
from .models import HandwrittenNotesDocument
from .ocr_utils import extract_text_from_image, get_gemini_cost_stats

logger = logging.getLogger(__name__)

from django.views.generic import TemplateView

from .models import UserSettings, COLOR_SCHEME_CHOICES, FONT_FAMILY_CHOICES

@require_http_methods(["GET", "POST"])
@login_required
def user_settings_api(request):
    """
    GET:  return {"color_scheme": ..., "font_family": ...}
    POST: accept JSON {"color_scheme": ..., "font_family": ...},
          validate, save, and return same JSON.
    """
    # helper dicts for validation
    valid_color_schemes = {key for key, _ in COLOR_SCHEME_CHOICES}
    valid_font_families = {key for key, _ in FONT_FAMILY_CHOICES}

    # fetch or create
    settings_obj, _ = UserSettings.objects.get_or_create(user=request.user)

    if request.method == "GET":
        return JsonResponse({
            "color_scheme": settings_obj.color_scheme,
            "font_family": settings_obj.font_family,
        })

    # POST: parse and validate
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    color = data.get("color_scheme")
    font  = data.get("font_family")

    errors = {}
    if color not in valid_color_schemes:
        errors["color_scheme"] = "Invalid choice"
    if font not in valid_font_families:
        errors["font_family"] = "Invalid choice"
    if errors:
        return JsonResponse({"errors": errors}, status=400)

    settings_obj.color_scheme = color
    settings_obj.font_family  = font
    settings_obj.save()

    return JsonResponse({
        "color_scheme": settings_obj.color_scheme,
        "font_family": settings_obj.font_family,
    })
    
class UserSettingsPageView(TemplateView):
    template_name = "aquillm/user_settings.html"
    

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
            error_message = "Invalid form submisison"
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


# def llm_convo(request):


#     reranked_results = []
#     error_message = None
#     llm_response = None
#     if request.method == 'POST':
#         form = SearchForm(request.user, request.POST)
#         if form.is_valid():
#             query = form.cleaned_data['query']
#             top_k = form.cleaned_data['top_k']
#             collections = form.cleaned_data['collections']
#             searchable_docs = get_user_accessible_documents(request.user, collections=collections)
#             _, _, reranked_results = text_chunk_search(query, top_k, searchable_docs)
#         else:
#             error_message = "Invalid form submisison"
#     else:
#         form = SearchForm(request.user)

#     context = {
#         'form': form,
#         'reranked_results': reranked_results,
#         'vector_results': vector_results,
#         'trigram_results': trigram_results,
#         'error_message': error_message
#     }

#     return render(request, 'aquillm/llm_convo.html', context)


# helper func, not a view
def get_doc(request, doc_id):
    doc = None
    for t in DESCENDED_FROM_DOCUMENT:
        doc = t.objects.filter(id=doc_id).first()
        if doc:
            break
    if not doc:
        raise Http404("Requested document does not exist")
    if not doc.collection.user_can_view(request.user):
        raise PermissionDenied("You don't have access to the collection containing this document")
    return doc

@require_http_methods(['GET'])
@login_required
def pdf(request, doc_id):
    doc = get_doc(request, doc_id)
    if doc.pdf_file:
        response = HttpResponse(doc.pdf_file, content_type='application/pdf')
        return response
    else:
        raise Http404("Requested document does not have an associated PDF")
    
    
@require_http_methods(['GET'])
@login_required
def document(request, doc_id):
    doc = get_doc(request, doc_id)
    context = {'document': doc}
    return render(request, 'aquillm/document.html', context)


# helper func, not a view
def insert_one_from_arxiv(arxiv_id, collection, user):
    status_message = ""
    tex_req = requests.get('https://arxiv.org/src/' + arxiv_id)
    pdf_req = requests.get('https://arxiv.org/pdf/' + arxiv_id)
    metadata_req = requests.get('http://export.arxiv.org/api/query?id_list=' + arxiv_id)
    if metadata_req.status_code == 404 or (tex_req.status_code == 404 and pdf_req.status_code == 404):
        status_message += "ERROR: 404 from ArXiv, is the DOI correct?"
    elif tex_req.status_code not in [200, 404] or pdf_req.status_code not in [200, 404] or metadata_req.status_code not in [200, 404]:
        error_str = f"ERROR -- DOI {arxiv_id}: LaTeX status code {tex_req.status_code}, PDF status code {pdf_req.status_code}, metadata status code {metadata_req.status_code}"
        logger.error(error_str)
        status_message += error_str
    else:
        xmldoc = minidom.parseString(metadata_req.content)
        title = ' '.join(xmldoc.getElementsByTagName('entry')[0].getElementsByTagName('title')[0].firstChild.data.split()) # type: ignore
        if tex_req.status_code == 200:
            status_message += f"Got LaTeX source for {arxiv_id}\n"
            tgz_io = io.BytesIO(tex_req.content)
            tex_str = ""
            with gzip.open(tgz_io, 'rb') as gz:
                with tarfile.open(fileobj=gz) as tar: # type: ignore
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith('.tex'):
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode('utf-8')
                                tex_str += content + '\n\n'
            doc = TeXDocument(
                collection = collection,
                title = title,
                full_text = tex_str,
                ingested_by=user
            )
            if pdf_req.status_code == 200:
                status_message += f'Got PDF for {arxiv_id}\n'
                doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
            doc.save()
        elif pdf_req.status_code == 200:
            status_message += f'Got PDF for {arxiv_id}\n'
            doc = PDFDocument(
                collection = collection,
                title = title,
                ingested_by=user
            )
            doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
            doc.save()
    return status_message



@require_http_methods(['GET', 'POST'])
@login_required
def insert_arxiv(request):
    status_message = None
    if request.method == 'POST':
        form = ArXiVForm(request.user, request.POST)
        if form.is_valid():
            arxiv_id = re.sub(r'[^\d.]', '', form.cleaned_data['arxiv_id']).lstrip('.')
            collection = Collection.objects.get(id=form.cleaned_data['collection'])
            status_message = insert_one_from_arxiv(arxiv_id, collection, request.user)
    else:
        form = ArXiVForm(request.user)

    context = {
        'status_message' : status_message,
        'form' : form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Return only the partial template for the modal (without the full page chrome)
        return render(request, 'aquillm/insert_arxiv_modal.html', context)

    return render(request, 'aquillm/insert_arxiv.html', context)


#TODO: make this much nicer
@require_http_methods(['GET', 'POST'])
@login_required
def user_collections(request):
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

# @requires_csrf_token
# @require_http_methods(['GET'])
# @login_required
# def get_collections_json(request):
#     colperms = CollectionPermission.objects.filter(user=request.user)
#     return JsonResponse({"collections": [{'id': colperm.collection.id,
#                                           'name': colperm.collection.name,
#                                           'document_count': len(colperm.collection.documents),
#                                           'permission': colperm.permission} for colperm in colperms]})



@require_http_methods(['GET'])
@login_required
def collection(request, col_id):
    """View to display a collection and its contents"""
    try:
        collection = get_object_or_404(Collection, pk=col_id)
        if not collection.user_can_view(request.user):
            return HttpResponseForbidden("User does not have permission to view this collection.")
        
        # Get all collections the user can edit for move functionality
        available_collections = Collection.objects.filter_by_user_perm(request.user, 'EDIT')
        
        # Return HTML template for browser requests
        return render(request, 'aquillm/collection.html', {
            'collection': collection,
            'path': collection.get_path(),
            'can_edit': collection.user_can_edit(request.user),
            'can_delete': collection.user_can_manage(request.user),
            'available_collections': available_collections,
        })
    except Exception as e:
        logger.error(f"Error in collection view: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(['GET', 'POST'])
@login_required
def ingest_pdf(request):
    status_message = None
    if request.method == 'POST':
        form = PDFDocumentForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf_file']
            collection = form.cleaned_data['collection']
            title = form.cleaned_data['title'].strip()
            PDFDocument(title=title, pdf_file=pdf_file, collection=collection, ingested_by=request.user).save()
            status_message = "Ingestion Started"
        else:
            status_message = "Invalid Form Input"
    else:
        form = PDFDocumentForm(request.user)
    
    context = {
        'status_message' : status_message,
        'form' : form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Return only the partial template for the modal
        return render(request, 'aquillm/ingest_pdf_modal.html', context)

    return render(request, 'aquillm/ingest_pdf.html', context)

@require_http_methods(['GET', 'POST'])
@login_required
def ingest_vtt(request):
    status_message = None
    if request.method == 'POST':
        form = VTTDocumentForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.cleaned_data['audio_file']
            vtt_file = form.cleaned_data['vtt_file']
            title = form.cleaned_data['title'].strip()
            collection = form.cleaned_data['collection']
            full_text = vtt.to_text(vtt.coalesce_captions(vtt.parse(vtt_file), max_gap=20.0, max_size=1024))
            VTTDocument(title=title,
                        audio_file=audio_file,
                        full_text=full_text,
                        collection=collection,
                        ingested_by=request.user).save()
            status_message = 'Success'
        else:
            status_message = 'Invalid Form Input'
    else:
        form = VTTDocumentForm(request.user)

    context = {
        'status_message' : status_message,
        'form' : form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Return only the partial template for the modal (without the full page chrome)
        return render(request, 'aquillm/ingest_vtt_modal.html', context)
    
    return render(request, 'aquillm/ingest_vtt.html', context)


###
###
###*********************************************###
# Ingest Notes Function
@require_http_methods(['GET', 'POST'])
@login_required
def ingest_handwritten_notes(request):
    """
    View for handling the upload and processing of handwritten notes.
    
    This view performs the following:
    1. Handles the form submission for handwritten notes
    2. Validates and processes the uploaded image
    3. Creates a HandwrittenNotesDocument with the extracted text
    4. Handles LaTeX conversion if requested
    """
    status_message = None
    if request.method == 'POST':
        form = HandwrittenNotesForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image_file']
            title = form.cleaned_data['title'].strip()
            collection = form.cleaned_data['collection']
            convert_to_latex = form.cleaned_data.get('convert_to_latex', False)

            # Process the image file and create document
            try:
                # Verify the image file is valid
                if not image_file or not hasattr(image_file, 'size') or image_file.size == 0:
                    raise ValueError("Invalid or empty image file")
                
                # Reset file pointer to ensure we read from the beginning
                image_file.seek(0)
                
                # Process the image and extract text using the model's own logic
                document = HandwrittenNotesDocument(
                    title=title,
                    image_file=image_file,
                    collection=collection,
                    ingested_by=request.user,
                    convert_to_latex=convert_to_latex,  # Pass the LaTeX flag to the model
                )
                
                # Save the document, which will trigger text extraction
                document.save()
                
                status_message = 'Success'
                    
            except Exception as e:
                status_message = f'Error: {str(e)}'
        else:
            status_message = 'Invalid Form Input'
    else:
        form = HandwrittenNotesForm(request.user)
    
    context = {
        'form': form,
        'status_message': status_message
    }
    
    # Handle AJAX requests for React integration
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        if status_message == 'Success':
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'status': 'error', 'error': status_message}, status=400)
    
    # Regular form submission
    return render(request, 'aquillm/ingest_handwritten_notes.html', context)


@require_http_methods(['DELETE'])
def delete_document(request, doc_id):
    doc = get_doc(request, doc_id)
    if not doc.collection.user_can_edit(request.user):
        return HttpResponseForbidden("User does not have permission to delete this item.")
    doc.delete()
    return HttpResponse(status=200)

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
@require_http_methods(['GET'])
def health_check(request):
    return HttpResponse(status=200)

@require_http_methods(['GET'])
@login_required
def user_ws_convos(request):
    convos = WSConversation.objects.filter(owner=request.user).order_by('-created_at') # this used to be updated-at
    return render(request, 'aquillm/user_ws_convos.html', {'conversations': convos})


# breaks for local debugging with all model objects in scope. 
# also good to have a hardcoded breakpoint, because the debugger won't attach if no breakpoints are set.
if DEBUG:
    @require_http_methods(['GET'])
    @login_required
    def debug_models(request):
        models = apps.get_models()
        model_instances = {model.__name__ : list(model.objects.all()) for model in models}
        breakpoint()
        return HttpResponse(status=200)

@login_required
@require_http_methods(['GET'])
def ingestion_monitor(request):
    in_progress = PDFDocument.objects.filter(ingestion_complete=False, ingested_by=request.user)
    protocol = 'wss://' if request.is_secure() else 'ws://'
    host = request.get_host()
    return JsonResponse([{"documentName": doc.title,
                          "documentId": doc.id,
                          "websocketUrl": protocol + host + "/ingest/monitor/" + doc.id + "/"}
                          for doc in in_progress])

@login_required
@require_http_methods(['GET'])
def ingestion_dashboard(request):
    return render(request, 'aquillm/ingestion_dashboard.html')


@login_required
@require_http_methods(['GET'])
def pdf_ingestion_monitor(request, doc_id):
    return render(request, 'aquillm/pdf_ingestion_monitor.html', {'doc_id': doc_id})

@login_required
@require_http_methods(['GET'])
def gemini_cost_monitor(request):
    """View to display the current Gemini API cost statistics"""
    stats = get_gemini_cost_stats()
    return render(request, 'aquillm/gemini_cost_monitor.html', {'stats': stats})

@login_required
@require_http_methods(['GET'])
def email_whitelist(request):
    return render(request, 'aquillm/email_whitelist.html')

urlpatterns = [
    path("search/", search, name="search"),
    path("insert_arxiv/", insert_arxiv, name="insert_arxiv"),
    path("pdf/<uuid:doc_id>/", pdf, name="pdf"),
    path("document/<uuid:doc_id>/", document, name="document"),
    path("user_collections/", user_collections, name="user_collections"),
    path("collection/<int:col_id>/", collection, name="collection"),
    path("ingest_pdf/", ingest_pdf, name="ingest_pdf"),
    path("ingest_vtt/", ingest_vtt, name="ingest_vtt"),
    path("user_ws_convos/", user_ws_convos, name="user_ws_convos"),
    path("react_test", react_test, name="react_test"),
    path("pdf_ingestion_monitor/<int:doc_id>/", pdf_ingestion_monitor, name="pdf_ingestion_monitor"),
    path("ingestion_dashboard/", ingestion_dashboard, name="ingestion_dashboard"),
    path("email_whitelist/", email_whitelist, name="email_whitelist"),
    path("ingest_handwritten_notes/", ingest_handwritten_notes, name="ingest_handwritten_notes"),
    path('gemini-costs/', gemini_cost_monitor, name='gemini_cost_monitor')
]