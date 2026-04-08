"""Page views for document ingestion."""
import gzip
import io
import structlog
import re
import tarfile
from xml.dom import minidom

import requests
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.db import DatabaseError
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from apps.collections.models import Collection
from apps.documents.models import (
    PDFDocument,
    TeXDocument,
    VTTDocument,
    HandwrittenNotesDocument,
    DESCENDED_FROM_DOCUMENT,
    document_modality,
    document_has_raw_media,
    document_provider_name,
    document_provider_model,
)
from aquillm.forms import (
    ArXiVForm,
    PDFDocumentForm,
    VTTDocumentForm,
    HandwrittenNotesForm,
)
from aquillm import vtt

logger = structlog.stdlib.get_logger(__name__)


def insert_one_from_arxiv(arxiv_id, collection, user):
    """Helper function to ingest a paper from arXiv (form-based version)."""
    status_message = ""
    tex_req = requests.get('https://arxiv.org/src/' + arxiv_id)
    pdf_req = requests.get('https://arxiv.org/pdf/' + arxiv_id)
    metadata_req = requests.get('http://export.arxiv.org/api/query?id_list=' + arxiv_id)
    if metadata_req.status_code == 404 or (tex_req.status_code == 404 and pdf_req.status_code == 404):
        status_message += "ERROR: 404 from ArXiv, is the DOI correct?"
    elif tex_req.status_code not in [200, 404] or pdf_req.status_code not in [200, 404] or metadata_req.status_code not in [200, 404]:
        error_str = f"ERROR -- DOI {arxiv_id}: LaTeX status code {tex_req.status_code}, PDF status code {pdf_req.status_code}, metadata status code {metadata_req.status_code}"
        logger.error("obs.ingest.view_error", arxiv_id=arxiv_id, tex_status=tex_req.status_code, pdf_status=pdf_req.status_code, metadata_status=metadata_req.status_code)
        status_message += error_str
    else:
        xmldoc = minidom.parseString(metadata_req.content)
        title = ' '.join(xmldoc.getElementsByTagName('entry')[0].getElementsByTagName('title')[0].firstChild.data.split())  # type: ignore
        if tex_req.status_code == 200:
            status_message += f"Got LaTeX source for {arxiv_id}\n"
            tgz_io = io.BytesIO(tex_req.content)
            tex_str = ""
            with gzip.open(tgz_io, 'rb') as gz:
                with tarfile.open(fileobj=gz) as tar:  # type: ignore
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith('.tex'):
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode('utf-8')
                                tex_str += content + '\n\n'
            doc = TeXDocument(
                collection=collection,
                title=title,
                full_text=tex_str,
                ingested_by=user
            )
            if pdf_req.status_code == 200:
                status_message += f'Got PDF for {arxiv_id}\n'
                doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
            doc.save()
        elif pdf_req.status_code == 200:
            status_message += f'Got PDF for {arxiv_id}\n'
            doc = PDFDocument(
                collection=collection,
                title=title,
                ingested_by=user
            )
            doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
            doc.save()
    return status_message


@require_http_methods(['GET', 'POST'])
@login_required
def insert_arxiv(request):
    """Form-based ArXiv paper ingestion."""
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
        'status_message': status_message,
        'form': form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'aquillm/insert_arxiv_modal.html', context)

    return render(request, 'aquillm/insert_arxiv.html', context)


@require_http_methods(['GET', 'POST'])
@login_required
def ingest_pdf(request):
    """Form-based PDF ingestion."""
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
        'status_message': status_message,
        'form': form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'aquillm/ingest_pdf_modal.html', context)

    return render(request, 'aquillm/ingest_pdf.html', context)


@require_http_methods(['GET', 'POST'])
@login_required
def ingest_vtt(request):
    """Form-based VTT (caption) ingestion."""
    status_message = None
    if request.method == 'POST':
        form = VTTDocumentForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.cleaned_data['audio_file']
            vtt_file = form.cleaned_data['vtt_file']
            title = form.cleaned_data['title'].strip()
            collection = form.cleaned_data['collection']
            full_text = vtt.to_text(vtt.coalesce_captions(vtt.parse(vtt_file), max_gap=20.0, max_size=1024))
            VTTDocument(
                title=title,
                audio_file=audio_file,
                full_text=full_text,
                collection=collection,
                ingested_by=request.user
            ).save()
            status_message = 'Success'
        else:
            status_message = 'Invalid Form Input'
    else:
        form = VTTDocumentForm(request.user)

    context = {
        'status_message': status_message,
        'form': form
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'aquillm/ingest_vtt_modal.html', context)
    
    return render(request, 'aquillm/ingest_vtt.html', context)


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

            try:
                if not image_file or not hasattr(image_file, 'size') or image_file.size == 0:
                    raise ValueError("Invalid or empty image file")
                
                image_file.seek(0)
                
                document = HandwrittenNotesDocument(
                    title=title,
                    image_file=image_file,
                    collection=collection,
                    ingested_by=request.user,
                    convert_to_latex=convert_to_latex,
                )
                
                document.save()
                status_message = 'Success'
                    
            except Exception as e:
                status_message = f'Error: {str(e)}'
        else:
            form_errors = form.errors.get_json_data()
            status_message = f'Invalid Form Input: {form_errors}'
    else:
        form = HandwrittenNotesForm(request.user)
    
    context = {
        'form': form,
        'status_message': status_message
    }
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        if status_message == 'Success':
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'status': 'error', 'error': status_message}, status=400)
    
    return render(request, 'aquillm/ingest_handwritten_notes.html', context)


@login_required
@require_http_methods(['GET'])
def ingestion_monitor(request):
    """Get documents currently being ingested (page version returning JSON)."""
    in_progress = []
    for doc_model in DESCENDED_FROM_DOCUMENT:
        in_progress.extend(
            list(
                doc_model.objects.filter(ingestion_complete=False, ingested_by=request.user).only("id", "title")
            )
        )
    protocol = 'wss://' if request.is_secure() else 'ws://'
    host = request.get_host()
    return JsonResponse(
        [
            {
                "documentName": doc.title,
                "documentId": str(doc.id),
                "websocketUrl": protocol + host + "/ingest/monitor/" + str(doc.id) + "/",
                "modality": document_modality(doc),
                "rawMediaSaved": document_has_raw_media(doc),
                "textExtracted": bool((doc.full_text or "").strip()),
                "provider": document_provider_name(doc),
                "providerModel": document_provider_model(doc),
            }
            for doc in in_progress
        ],
        safe=False
    )


@login_required
@require_http_methods(['GET'])
def ingestion_dashboard(request):
    """Display the ingestion dashboard page."""
    return render(request, 'aquillm/ingestion_dashboard.html')


@login_required
@require_http_methods(['GET'])
def pdf_ingestion_monitor(request, doc_id):
    """Display the PDF ingestion monitor page."""
    return render(request, 'aquillm/pdf_ingestion_monitor.html', {'doc_id': doc_id})


__all__ = [
    'insert_one_from_arxiv',
    'insert_arxiv',
    'ingest_pdf',
    'ingest_vtt',
    'ingest_handwritten_notes',
    'ingestion_monitor',
    'ingestion_dashboard',
    'pdf_ingestion_monitor',
]
