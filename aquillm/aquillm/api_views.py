import requests
import logging
import gzip
import tarfile
import io
import chardet
import json
from uuid import UUID
from xml.dom import minidom
from django.urls import path, include
# Comment out or remove the top-level import if it exists:
# from trafilatura import fetch_url, extract, extract_metadata # Keep commented or remove if not used elsewhere
from django.core.files.base import ContentFile

# Import the new Celery task
from .crawler_tasks import crawl_and_ingest_webpage

from .vtt import parse, to_text, coalesce_captions
from .models import WSConversation, ConversationFile, Document, PDFDocument, TeXDocument, VTTDocument, Collection, CollectionPermission, EmailWhitelist, DESCENDED_FROM_DOCUMENT, DuplicateDocumentError, RawTextDocument
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model

from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, FileResponse

from django.shortcuts import get_object_or_404

from django.core.files.base import ContentFile
from django.core.validators import FileExtensionValidator, validate_email
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.db import DatabaseError, transaction
from django.db.models import Q
from aquillm.views import user_settings_api
from django.core.exceptions import PermissionDenied


logger = logging.getLogger(__name__)



# helper func, not a view
def insert_one_from_arxiv(arxiv_id, collection, user):

    def save_pdf_doc(content, title):
        doc = PDFDocument(
            collection=collection,
            title=title,
            ingested_by=user
        )
        doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(content), save=False)
        doc.save()

    status = {"message": "", "errors": []}
    tex_req = requests.get('https://arxiv.org/src/' + arxiv_id)
    pdf_req = requests.get('https://arxiv.org/pdf/' + arxiv_id)
    metadata_req = requests.get('http://export.arxiv.org/api/query?id_list=' + arxiv_id)

    if metadata_req.status_code == 404 or (tex_req.status_code == 404 and pdf_req.status_code == 404):
        status["errors"].append("ERROR: 404 from ArXiv, is the DOI correct?")
    elif (tex_req.status_code not in [200, 404] or 
          pdf_req.status_code not in [200, 404] or 
          metadata_req.status_code not in [200, 404]):
        error_str = (
            f"ERROR -- DOI {arxiv_id}: LaTeX status code {tex_req.status_code}, "
            f"PDF status code {pdf_req.status_code}, metadata status code {metadata_req.status_code}"
        )
        logger.error(error_str)
        status["errors"].append(error_str)
    else:
        # Parse metadata to extract the title.
        xmldoc = minidom.parseString(metadata_req.content)
        title = ' '.join(
            xmldoc.getElementsByTagName('entry')[0]
                 .getElementsByTagName('title')[0]
                 .firstChild.data.split() # type: ignore
        )
        
        # Process the /src/ endpoint if it returned 200.
        if tex_req.status_code == 200:
            # Check if the content appears to be a PDF.
            if tex_req.content.startswith(b'%PDF'):
                status["message"] += f"Got PDF for {arxiv_id}\n"
                save_pdf_doc(tex_req.content, title)
            else:
                status["message"] += f"Got LaTeX source for {arxiv_id}\n"
                tgz_io = io.BytesIO(tex_req.content)
                tex_str = ""
                # Extract the tar.gz archive containing the LaTeX source.
                with gzip.open(tgz_io, 'rb') as gz:
                    with tarfile.open(fileobj=gz) as tar: # type: ignore
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith('.tex'):
                                f = tar.extractfile(member)
                                if f:
                                    tex_bytes = f.read()
                                    encoding = chardet.detect(tex_bytes)['encoding']
                                    if not encoding:
                                        # been having problems with chardet not detecting ASCII correctly
                                        if not any(x > 127 for x in tex_bytes): # test for bytes with high bit set
                                            encoding = 'ascii'
                                        else:
                                            raise ValueError("Could not detect encoding of LaTeX source")
                                    content = tex_bytes.decode(encoding)
                                    tex_str += content + '\n\n'
                doc = TeXDocument(
                    collection=collection,
                    title=title,
                    full_text=tex_str,
                    ingested_by=user
                )
                # Optionally attach the PDF if available.
                if pdf_req.status_code == 200:
                    status["message"] += f"Got PDF for {arxiv_id}\n"
                    doc.pdf_file.save(f'arxiv:{arxiv_id}.pdf', ContentFile(pdf_req.content), save=False)
                doc.save()
        # If the /src/ endpoint didn't work but the PDF endpoint did, use that.
        elif pdf_req.status_code == 200:
            status["message"] += f"Got PDF for {arxiv_id}\n"
            save_pdf_doc(pdf_req.content, title)
            
    return status



@login_required
@require_http_methods(["POST"])
def ingest_arxiv(request):
    user = request.user
    arxiv_id = request.POST.get('arxiv_id')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not arxiv_id:
        return JsonResponse({'error': 'No arXiv ID provided'}, status=400)
    try:
        status = insert_one_from_arxiv(arxiv_id, collection, user)
        if status["errors"]:
            return JsonResponse({'error': status["errors"]}, status=500)
        return JsonResponse({'message': status["message"]})
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({'error': e.message}, status=400)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving document'}, status=500)

@login_required
@require_http_methods(["POST"])
def ingest_pdf(request):
    user = request.user
    pdf_file = request.FILES.get('pdf_file')
    title = request.POST.get('title')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not pdf_file:
        return JsonResponse({'error': 'No PDF file provided'}, status=400)
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)
    try:
        FileExtensionValidator(['pdf'])(pdf_file)
    except ValidationError as e:
        return JsonResponse({'error': 'Invalid file extension. Only PDF files are allowed.'}, status=400)
    doc = PDFDocument(
        collection = collection,
        title = title,
        ingested_by = user
    )
    doc.pdf_file = pdf_file
    try:
        doc.save()
    except DuplicateDocumentError as e:
        logger.error(e.message)
        return JsonResponse({'error', e.message}, status=200)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving PDFDocument'}, status=500)

    return JsonResponse({'status_message': 'Success'})

@login_required
@require_http_methods(["POST"])
def ingest_vtt(request):
    user = request.user
    vtt_file = request.FILES.get('vtt_file')
    audio_file = request.FILES.get('audio_file')
    title = request.POST.get('title')
    collection = Collection.objects.filter(pk=request.POST.get('collection')).first()
    if not collection or not collection.user_can_edit(user):
        return JsonResponse({'error': 'Collection does not exist, was not provided, or user does not have permission to edit this collection'}, status=403)
    if not vtt_file:
        return JsonResponse({'error': 'No VTT file provided'}, status=400)
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)
    try:
        FileExtensionValidator(['vtt'])(vtt_file)
    except ValidationError as e:
        return JsonResponse({'error': 'Invalid file extension. Only VTT files are allowed.'}, status=400)
    full_text = to_text(coalesce_captions(parse(vtt_file)))
    
    doc = VTTDocument(
        collection = collection,
        title = title,
        ingested_by = user,
        full_text = full_text
    )
    if audio_file:
        doc.audio_file = audio_file
    try:
        doc.save()
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        return JsonResponse({'error': 'Database error occurred while saving VTTDocument'}, status=500)

    return JsonResponse({'status_message': 'Success'})


@login_required
@require_http_methods(["DELETE"])
def delete_collection(request, collection_id):
    user = request.user
    collection = get_object_or_404(Collection, id=collection_id)
    
    # Check if user has MANAGE permission for this collection
    if not collection.user_can_manage(user):
        return JsonResponse({'error': 'You do not have permission to delete this collection'}, status=403)
    
    # Get children before deletion for notification purposes
    children_count = collection.children.count()
    documents_count = sum(len(x.objects.filter(collection=collection)) for x in DESCENDED_FROM_DOCUMENT)
    
    try:
        # Django will cascade delete children collections and documents
        collection.delete()
        return JsonResponse({
            'success': True,
            'message': f'Collection deleted successfully along with {children_count} subcollections and {documents_count} documents'
        })
    except Exception as e:
        logger.error(f"Error deleting collection {collection_id}: {e}")
        return JsonResponse({'error': f'Failed to delete collection: {str(e)}'}, status=500)

@login_required
@require_http_methods(["DELETE"])
def delete_document(request, doc_id):
    user = request.user
    
    # Try to find the document among all document types
    document = Document.get_by_id(doc_id)
    
    if not document:
        return JsonResponse({'error': 'Document not found'}, status=404)
    
    title = document.title
    # Check if user has EDIT permission for the document's collection
    if not document.collection.user_can_edit(user):
        return JsonResponse({'error': 'You do not have permission to delete this document'}, status=403)
    
    try:
        document.delete()
        return JsonResponse({
            'success': True,
            'message': f'{title} deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        return JsonResponse({'error': f'Failed to delete document: {str(e)}'}, status=500)


@require_http_methods(['GET', 'POST'])
@login_required
def collections(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        if not name:
            return JsonResponse({'error': 'Name is required'}, status=400)

        with transaction.atomic():
            collection = Collection.objects.create(name=name)
            CollectionPermission.objects.create(
                collection=collection,
                user=request.user,
                permission='MANAGE'
            )

            # Handle additional permissions
            for viewer in data.get('viewers', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=viewer),
                    permission='VIEW'
                )
            for editor in data.get('editors', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=editor),
                    permission='EDIT'
                )
            for admin in data.get('admins', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=admin),
                    permission='MANAGE'
                )

            return JsonResponse({
                'id': collection.id,
                'name': collection.name,
                'parent': collection.parent.id if collection.parent else None,
                'path': collection.get_path(),
                'document_count': len(collection.documents),
                'children_count': collection.children.count(),
                'permission': 'MANAGE'
            })

    # For GET requests, get all collections where the user has any permission
    colperms = CollectionPermission.objects.filter(user=request.user)
    collections = []
    for colperm in colperms:
        collections.append({
            'id': colperm.collection.id,
            'name': colperm.collection.name,
            'parent': colperm.collection.parent.id if colperm.collection.parent else None,
            'path': colperm.collection.get_path(),
            'document_count': len(colperm.collection.documents),
            'children_count': colperm.collection.children.count(),
            'permission': colperm.permission
        })
    return JsonResponse({"collections": collections})


@require_http_methods(["POST"])
@login_required
def move_collection(request, collection_id):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    new_parent_id = data.get("new_parent_id")  # Can be None

    try:
        collection = Collection.objects.get(id=collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Collection not found"}, status=404)

    # Check that the user has permission to manage (move) this collection.
    if not collection.user_can_manage(request.user):
        return JsonResponse({"error": "You do not have permission to move this collection"}, status=403)

    # If a new parent is provided, fetch it.
    new_parent = None
    if new_parent_id:
        try:
            new_parent = Collection.objects.get(id=new_parent_id)
        except Collection.DoesNotExist:
            return JsonResponse({"error": "Target parent collection not found"}, status=404)

    try:
        collection.move_to(new_parent=new_parent)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "message": "Collection moved successfully",
        "collection": {
            "id": collection.id,
            "name": collection.name,
            "parent": collection.parent.id if collection.parent else None,
            "path": collection.get_path(),
        }
    })

@require_http_methods(["POST"])
@login_required
def move_document(request, doc_id):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    new_collection_id = data.get("new_collection_id")
    if new_collection_id is None:
        return JsonResponse({"error": "new_collection_id is required"}, status=400)

    document = Document.get_by_id(doc_id)
    if not document:
        return JsonResponse({"error": "Document not found"}, status=404)

    try:
        new_collection = Collection.objects.get(id=new_collection_id)
    except Collection.DoesNotExist:
        return JsonResponse({"error": "Target collection not found"}, status=404)

    try:
        # Call the move_to method on the document
        document.move_to(new_collection)
    except ValidationError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "message": "Document moved successfully",
        "document": {
            "id": str(document.id),
            "title": document.title,
            "collection": new_collection.id,
        }
    })

@require_http_methods(['GET', 'POST'])
@login_required
def collection_permissions(request, col_id):
    collection = get_object_or_404(Collection, pk=col_id)
    User = get_user_model() # Ensure User model is available

    def serialize_user_for_permissions(user_obj):
        full_name = f"{user_obj.first_name} {user_obj.last_name}".strip()
        return {
            'id': user_obj.id,
            'username': user_obj.username,
            'email': user_obj.email,
            'full_name': full_name if full_name else user_obj.username
        }

    if request.method == 'GET':
        if not collection.user_can_manage(request.user):
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        view_permissions = CollectionPermission.objects.filter(collection=collection, permission='VIEW').select_related('user')
        edit_permissions = CollectionPermission.objects.filter(collection=collection, permission='EDIT').select_related('user')
        manage_permissions = CollectionPermission.objects.filter(collection=collection, permission='MANAGE').select_related('user')

        viewers = [serialize_user_for_permissions(perm.user) for perm in view_permissions]
        editors = [serialize_user_for_permissions(perm.user) for perm in edit_permissions]
        admins = [serialize_user_for_permissions(perm.user) for perm in manage_permissions]
        
        return JsonResponse({
            'viewers': viewers,
            'editors': editors,
            'admins': admins,
        })

    if not collection.user_can_manage(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    try:
        data = json.loads(request.body)
        with transaction.atomic():
            # Remove all existing permissions except the owner's
            CollectionPermission.objects.filter(collection=collection).exclude(user=request.user).delete()

            # Add new permissions
            for viewer in data.get('viewers', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=viewer),
                    permission='VIEW'
                )
            for editor in data.get('editors', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=editor),
                    permission='EDIT'
                )
            for admin in data.get('admins', []):
                CollectionPermission.objects.create(
                    collection=collection,
                    user=get_user_model().objects.get(id=admin),
                    permission='MANAGE'
                )

        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
@require_http_methods(['GET'])
def collection(request, col_id):
    try:
        collection = get_object_or_404(Collection, pk=col_id)
        if not collection.user_can_view(request.user):
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        # Get documents from all document types
        documents = []
        for model in DESCENDED_FROM_DOCUMENT:
            docs = model.objects.filter(collection=collection)
            for doc in docs:
                documents.append({
                    'id': str(doc.id),
                    'title': getattr(doc, 'title', None) or getattr(doc, 'name', 'Untitled'),
                    'type': doc.__class__.__name__,
                    'ingestion_date': doc.ingestion_date.isoformat() if hasattr(doc, 'ingestion_date') and doc.ingestion_date else None,
                })

        # Get child collections
        children = [{
            'id': child.id,
            'name': child.name,
            'document_count': len([doc for doc in child.documents]) if hasattr(child, 'documents') else 0,
            'created_at': child.created_at.isoformat() if hasattr(child, 'created_at') and child.created_at else None,
        } for child in collection.children.all()]

        response_data = {
            'collection': {
                'id': collection.id,
                'name': collection.name,
                'path': collection.get_path(),
                'parent': collection.parent.id if collection.parent else None,
                'created_at': collection.created_at.isoformat() if hasattr(collection, 'created_at') and collection.created_at else None,
                'updated_at': collection.updated_at.isoformat() if hasattr(collection, 'updated_at') and collection.updated_at else None,
            },
            'documents': documents,
            'children': children,
            'can_edit': collection.user_can_edit(request.user),
            'can_manage': collection.user_can_manage(request.user),
        }
        return JsonResponse(response_data)
    except Exception as e:
        logger.error(f"Error processing collection data: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

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
def search_users(request):
    query = request.GET.get('query', '').strip()
    exclude_current = request.GET.get('exclude_current', 'false').lower() == 'true'
    
    if not query:
        return JsonResponse({'users': []})

    User = get_user_model()
    users = User.objects.filter(Q(email__icontains=query) | Q(username__icontains=query) | Q(first_name__icontains=query) | Q(last_name__icontains=query)).distinct()

    if exclude_current:
        users = users.exclude(id=request.user.id)

    # Create a list with calculated full_name
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
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return JsonResponse({'whitelisted': list(EmailWhitelist.objects.all().values_list('email', flat=True))})

@login_required
@require_http_methods(['POST', 'DELETE'])
def whitelisted_email(request, email):
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
    # unreachable, just keeping the type checker happy
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@login_required
@require_http_methods(["POST"])
def ingest_webpage(request):
    """
    API endpoint to initiate asynchronous webpage crawling and ingestion.
    """
    try:
        data = json.loads(request.body)
        url = data.get('url')
        collection_id = data.get('collection_id')
        # Get depth, default to 1 if not provided or invalid type
        try:
            depth = int(data.get('depth', 1))
            if depth < 0:
                 depth = 0 # Ensure depth is non-negative (0 means initial page only)
        except (ValueError, TypeError):
            depth = 1 # Default to 1 if conversion fails

        if not url or not collection_id:
            logger.warning("Ingest webpage request missing url or collection_id.")
            return JsonResponse({'error': 'Missing url or collection_id'}, status=400)

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
             logger.warning(f"Invalid URL format received: {url}")
             return JsonResponse({'error': 'Invalid URL format. Must start with http:// or https://'}, status=400)

        # Use try-except for fetching collection to handle 404 within the main try block
        try:
            collection = Collection.objects.get(pk=collection_id)
        except Collection.DoesNotExist:
             logger.warning(f"Attempted ingest to non-existent collection ID: {collection_id}")
             return JsonResponse({'error': 'Collection not found'}, status=404)

        # Permission Check
        if not collection.user_can_edit(request.user):
            logger.warning(f"Permission denied for user {request.user.id} on collection {collection_id} during webpage ingest.")
            # Raise PermissionDenied to be caught by the outer exception handler
            raise PermissionDenied("You do not have permission to add documents to this collection.")

        # --- Delegate to Celery Task ---
        try:
            logger.info(f"Dispatching crawl_and_ingest_webpage task for URL: {url}, Collection: {collection_id}, User: {request.user.id}, Depth: {depth}")
            # Call the Celery task asynchronously, passing the validated depth.
            crawl_and_ingest_webpage.delay(url, collection_id, request.user.id, max_depth=depth)

            # Return 202 Accepted immediately
            return JsonResponse({'message': 'Webpage crawl initiated successfully.'}, status=202)

        except Exception as task_dispatch_error:
             # Catch potential errors during task dispatch (e.g., Celery broker connection issues)
             logger.error(f"Failed to dispatch crawl_and_ingest_webpage task for URL {url}: {task_dispatch_error}", exc_info=True)
             return JsonResponse({'error': 'Failed to initiate webpage crawl task.'}, status=500)

    except json.JSONDecodeError:
        logger.warning("Received invalid JSON in ingest_webpage request.")
        return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
    # Note: ObjectDoesNotExist for Collection is handled inside the main try block now
    except PermissionDenied as e:
        # This will catch the PermissionDenied raised earlier
        return JsonResponse({'error': str(e)}, status=403)
    except Exception as e:
        # Generic catch-all for unexpected errors in the view setup phase
        logger.error(f"Unexpected error in ingest_webpage view setup: {e}", exc_info=True)
        return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)


@login_required
@require_http_methods(['GET'])
def conversation_file(request, convo_file_id):
    convo_file = get_object_or_404(ConversationFile, pk=convo_file_id)
    if not convo_file.conversation.owner == request.user:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    return FileResponse(convo_file.file, as_attachment=True)

urlpatterns = [
    path("collections/", collections, name="api_collections"),
    path("collection/<int:col_id>/", collection, name="api_collection"),
    path("collections/permissions/<int:col_id>/", collection_permissions, name="api_collection_permissions"),
    path("collections/move/<int:collection_id>/", move_collection, name="api_move_collection"),
    path("collections/delete/<int:collection_id>/", delete_collection, name="api_delete_collection"),
    path("ingest_arxiv/", ingest_arxiv, name="api_ingest_arxiv"),
    path("ingest_pdf/", ingest_pdf, name="api_ingest_pdf"),
    path("ingestion/monitor/", ingestion_monitor, name="api_ingestion_monitor"),
    path("documents/move/<uuid:doc_id>/", move_document, name="api_move_document"),
    path("documents/delete/<uuid:doc_id>/", delete_document, name="api_delete_document"),
    path("users/search/", search_users, name="api_search_users"),
    path("whitelisted_email/<str:email>/", whitelisted_email, name="api_whitelist_email"),
    path("whitelisted_emails/", whitelisted_emails, name="api_whitelist_emails"),
    path("ingest_vtt/", ingest_vtt, name="api_ingest_vtt"),
    path('user-settings/', user_settings_api, name='api-user-settings'),
    path('conversation_file/<int:convo_file_id>/', conversation_file, name='api_conversation_file'),
    # Add the new URL pattern for webpage ingestion:
    path("ingest_webpage/", ingest_webpage, name="api_ingest_webpage"),
]
