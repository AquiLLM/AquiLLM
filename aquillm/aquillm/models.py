from typing import Optional, Callable, Awaitable
from django.db import models, transaction
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.fields import ArrayField
from pgvector.django import VectorField, L2Distance, HnswIndex
from django.apps import apps
from django.core.exceptions import ValidationError, ObjectDoesNotExist
from django.db.models import Q

from tenacity import retry, wait_exponential
import uuid

from django.contrib.auth.models import User
from pypdf import PdfReader

import functools 

# for hashing full_text of documents to ensure unique contents
import hashlib

from django.template import Context
from django.core.serializers.json import DjangoJSONEncoder
import json
import logging
from django.db.models.query import QuerySet
from typing import  List, Type, Tuple
import time

from django.core.exceptions import ValidationError
from django.contrib.postgres.search import TrigramSimilarity
from django.core.validators import FileExtensionValidator
import concurrent.futures

from django.db import DatabaseError
from django.db.models import Case, When
from django.utils import timezone
from .utils import get_embedding
from .settings import BASE_DIR

logger = logging.getLogger(__name__)

from .celery import app
from celery.states import state, RECEIVED, STARTED, SUCCESS, FAILURE

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

channel_layer = get_channel_layer()
assert (channel_layer is not None and
    hasattr(channel_layer, 'group_send')) # keeps type checker happy

type DocumentChild = PDFDocument | TeXDocument | RawTextDocument | VTTDocument


COLOR_SCHEME_CHOICES = (
    ('aquillm_default_dark', 'Aquillm Default Dark'),
    ('aquillm_default_light', 'Aquillm Default Light'),
    ('aquillm_default_light_accessible_chat', 'Aquillm Default Light Accessible Chat'),
    ('aquillm_default_dark_accessible_chat', 'Aquillm Default Dark Accessible Chat'),
    ('high_contrast', 'High Contrast'),
)

FONT_FAMILY_CHOICES = (
    ('latin_modern_roman', 'Latin Modern Roman'),
    ('sans_serif', 'Sans-serif'),
    ('verdana', 'Verdana'),
    ('timesnewroman', 'Times New Roman'),
    ('opendyslexic', 'OpenDyslexic'),
    ('lexend', "Lexend"),
    ('comicsans', 'Comic Sans')
)

class UserSettings(models.Model):
    # OneToOneField ensures one settings record per user.
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    color_scheme = models.CharField(
        max_length=100,
        choices=COLOR_SCHEME_CHOICES,
        default='aquillm_default_dark'
    )
    font_family = models.CharField(
        max_length=50,
        choices=FONT_FAMILY_CHOICES,
        default='sans_serif'
    )

    def __str__(self):
        return f"{self.user.username}'s settings"


class ZoteroConnection(models.Model):
    """Stores Zotero OAuth credentials for a user"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='zotero_connection')
    api_key = models.CharField(max_length=255, help_text="Zotero API key from OAuth")
    zotero_user_id = models.CharField(max_length=100, help_text="Zotero user ID")
    connected_at = models.DateTimeField(auto_now_add=True)
    last_synced_at = models.DateTimeField(null=True, blank=True, help_text="Last time a sync was performed")

    def __str__(self):
        return f"{self.user.username}'s Zotero connection (User ID: {self.zotero_user_id})"

from .ocr_utils import extract_text_from_image

from django.core.files.storage import default_storage

class CollectionQuerySet(models.QuerySet):
    def filter_by_user_perm(self, user, perm='VIEW') -> 'CollectionQuerySet':
        perm_options = []
        if perm == 'VIEW':
            perm_options = ['VIEW', 'EDIT', 'MANAGE']
        elif perm == 'EDIT':
            perm_options = ['EDIT', 'MANAGE']
        elif perm == 'MANAGE':
            perm_options = ['MANAGE']
        else:
            raise ValueError(f"Invalid Permission type {perm}")

        return self.filter(id__in=[col_perm.collection.pk for col_perm in CollectionPermission.objects.filter(user=user, permission__in=perm_options)])


class Collection(models.Model):
    name = models.CharField(max_length=100)
    users = models.ManyToManyField(User, through='CollectionPermission')
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='children')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = CollectionQuerySet.as_manager()

    class Meta:
        unique_together = ('name', 'parent')
        ordering = ['name']

    def get_path(self):
        path = [self.name]
        current = self
        while current.parent:
            current = current.parent
            path.append(current.name)
        return '/'.join(reversed(path))

    def get_all_children(self):
        children = list(self.children.all()) # type: ignore
        for child in self.children.all(): # type: ignore
            children.extend(child.get_all_children())
        return children

    # returns a list of documents, not a queryset.
    @property
    def documents(self):
        return functools.reduce(lambda l, r: l + r, [list(x.objects.filter(collection=self)) for x in DESCENDED_FROM_DOCUMENT])

    def user_has_permission_in(self, user, permissions):
        # First check permissions directly on this collection
        if CollectionPermission.objects.filter(
            user=user,
            collection=self,
            permission__in=permissions
        ).exists():
            return True
        
        # If no direct permission, check parent collections recursively
        if self.parent:
            return self.parent.user_has_permission_in(user, permissions)
        
        return False
    

    def user_can_view(self, user):
        return self.user_has_permission_in(user, ['VIEW', 'EDIT', 'MANAGE'])
    
    def user_can_edit(self, user):
        return self.user_has_permission_in(user, ['EDIT', 'MANAGE'])
    
    def user_can_manage(self, user):
        return self.user_has_permission_in(user, ['MANAGE'])
    
    # returns a list of documents, not a queryset.
    @classmethod
    def get_user_accessible_documents(cls, user, collections: Optional[CollectionQuerySet]=None, perm='VIEW'):
        if collections is None:
            collections = cls.objects.all() # type: ignore
        # pylance doesn't understand custom querysets
        collections = collections.filter_by_user_perm(user, perm) # type: ignore
        documents = functools.reduce(lambda l, r: l + r, [list(x.objects.filter(collection__in=collections)) for x in DESCENDED_FROM_DOCUMENT])
        return documents

    def move_to(self, new_parent=None):
        """Move this collection to a new parent"""
        if new_parent and new_parent.id == self.pk:
            raise ValidationError("Cannot move a collection to itself")
        
        # Check for circular reference
        if new_parent:
            parent_check = new_parent
            while parent_check is not None:
                if parent_check.id == self.pk:
                    raise ValidationError("Cannot create circular reference in collection hierarchy")
                parent_check = parent_check.parent
        
        self.parent = new_parent
        self.save()

    def __str__(self):
        return f'{self.name}'
    
    def is_owner(self, user):
        """
        Check if the user is the owner (creator) of this collection.
        The owner is defined as the first user who was granted MANAGE permission.
        """
        # Get the earliest MANAGE permission for this collection
        earliest_manage_perm = CollectionPermission.objects.filter(
            collection=self,
            permission='MANAGE'
        ).order_by('id').first()
        
        if earliest_manage_perm:
            return earliest_manage_perm.user == user
        return False

    def get_user_permission_source(self, user):
        """
        Returns the collection where the user's permission is coming from.
        Useful for debugging permission issues with nested collections.
        
        Returns tuple: (source_collection, permission_level)
        If no permission found, returns (None, None)
        """
        # Check direct permissions
        permission = CollectionPermission.objects.filter(user=user, collection=self).first()
        if permission:
            return (self, permission.permission)
        
        # Check parent permissions
        if self.parent:
            return self.parent.get_user_permission_source(user)
            
        return (None, None)


class CollectionPermission(models.Model):
    PERMISSION_CHOICES = [
        ('VIEW', 'View'),
        ('EDIT', 'Edit'),
        ('MANAGE', 'Manage')
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    permission = models.CharField(max_length=10, choices=PERMISSION_CHOICES)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'collection'],
                name='unique_permission_constraint')
        ]

    def save(self, *args, **kwargs):
        with transaction.atomic():
            existing_permission = CollectionPermission.objects.filter(
                user=self.user,
                collection=self.collection
            ).first()

            if existing_permission and existing_permission.pk != self.pk:
                existing_permission.delete()

            super().save(*args, **kwargs)
            
@app.task(serializer='pickle', bind=True, track_started=True)
def create_chunks(self, doc_id:str): #naive method, just number of characters
    channel_layer = get_channel_layer()
    doc = Document.get_by_id(uuid.UUID(doc_id))
    if not doc:
        raise ObjectDoesNotExist(f"No document with id {doc_id}")
    try:
        async_to_sync(channel_layer.group_send)(f'ingestion-dashboard-{doc.ingested_by.id}', {
            'type': 'document.ingestion.start',
            'documentId': str(doc.id),
            'documentName': doc.title,
        })

        # Check if there's an existing document with the same full_text_hash
        existing_doc_with_same_hash = None
        for doc_type in DESCENDED_FROM_DOCUMENT:
            existing_doc_with_same_hash = doc_type.objects.filter(
                full_text_hash=doc.full_text_hash,
                ingestion_complete=True
            ).exclude(id=doc.id).first()
            if existing_doc_with_same_hash:
                break

        # If we found a duplicate, copy its chunks instead of regenerating
        if existing_doc_with_same_hash:
            logger.info(f"Found duplicate document {existing_doc_with_same_hash.id} with same content hash. Copying chunks...")
            existing_chunks = TextChunk.objects.filter(doc_id=existing_doc_with_same_hash.id)

            # Delete any existing chunks for this document
            TextChunk.objects.filter(doc_id=doc.id).delete()

            # Copy chunks from the existing document
            new_chunks = []
            for chunk in existing_chunks:
                new_chunks.append(TextChunk(
                    content=chunk.content,
                    start_position=chunk.start_position,
                    end_position=chunk.end_position,
                    doc_id=doc.id,
                    chunk_number=chunk.chunk_number,
                    embedding=chunk.embedding  # Reuse the same embedding
                ))

            # Bulk create the copied chunks
            TextChunk.objects.bulk_create(new_chunks)

            # Mark document as complete
            doc.ingestion_complete = True
            doc.save(dont_rechunk=True)

            async_to_sync(channel_layer.group_send)(f'document-ingest-{doc.id}', {
                'type': 'document.ingest.complete',
                'complete': True
            })

            logger.info(f"Copied {len(new_chunks)} chunks from document {existing_doc_with_same_hash.id} to {doc.id}")
            return

        # No duplicate found, create chunks normally
        chunk_size = apps.get_app_config('aquillm').chunk_size # type: ignore
        overlap = apps.get_app_config('aquillm').chunk_overlap # type: ignore
        chunk_pitch = chunk_size - overlap
        # Delete existing chunks for this document
        TextChunk.objects.filter(doc_id=doc.id).delete()
        last_character = len(doc.full_text) - 1
        # Create new chunks
        
        chunks = list([TextChunk(
                    content = doc.full_text[chunk_pitch * i : min((chunk_pitch * i) + chunk_size, last_character + 1)],
                    start_position=chunk_pitch * i,
                    end_position=min((chunk_pitch * i) + chunk_size, last_character + 1),
                    doc_id = doc.id,
                    chunk_number = i) for i in range(last_character // chunk_pitch + 1)])
        n_chunks = len(chunks)
        done_chunks = [0] # this has to be a list because of the way python handles closures

        def send_progress():
            done_chunks[0] += 1
            async_to_sync(channel_layer.group_send)(f'document-ingest-{doc.id}', {
                'type': 'document.ingest.progress',
                'progress': int((done_chunks[0] / n_chunks) * 100),
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as e:
            e.map(functools.partial(TextChunk.get_chunk_embedding, callback=send_progress), chunks)
        
        TextChunk.objects.bulk_create(chunks)
        doc.ingestion_complete = True
        doc.save(dont_rechunk=True)
        async_to_sync(channel_layer.group_send)(f'document-ingest-{doc.id}', {
            'type': 'document.ingest.complete',
            'complete' : True
        })
    except Exception as e:
        logger.error(f"Error creating chunks for document {doc.id}: {str(e)}")
        self.update_state(state=FAILURE)
        
        # TEMPORARY FIX: Don't delete documents on error
        # doc.delete()
        
        # Instead, mark the document as complete but with error status
        doc.ingestion_complete = True
        doc.full_text += f"\n\nERROR DURING PROCESSING: {str(e)}"
        doc.save(dont_rechunk=True)
        
        raise e    

class DuplicateDocumentError(ValidationError):
    def __init__(self, message):
        super().__init__(message)


class Document(models.Model):
    pkid = models.BigAutoField(primary_key=True, editable=False)
    id = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    title = models.CharField(max_length=200)
    full_text = models.TextField()
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name='%(class)s_documents')
    full_text_hash = models.CharField(max_length=64, db_index=True)
    ingested_by = models.ForeignKey(User, on_delete=models.RESTRICT)
    ingestion_date = models.DateTimeField(auto_now_add=True)
    ingestion_complete = models.BooleanField(default=True)
    class Meta:
        abstract = True
        constraints = [
            models.UniqueConstraint(
                fields=['collection', 'full_text_hash'],
                name='%(class)s_document_collection_unique'
            )
        ]
        ordering = ['-ingestion_date', 'title']

    @staticmethod
    def hash_fn(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @property
    def chunks(self):
        return TextChunk.objects.filter(doc_id=self.id)

    @staticmethod
    def filter(*args, **kwargs) -> List[DocumentChild]:
        return functools.reduce(lambda l, r: l + r, [list(x.objects.filter(*args, **kwargs)) for x in DESCENDED_FROM_DOCUMENT])

    @staticmethod
    def get_by_id(doc_id: uuid.UUID) -> Optional[DocumentChild]:
        for t in DESCENDED_FROM_DOCUMENT:
            doc = t.objects.filter(id=doc_id).first()
            if doc:
                return doc
        return None

    
    def save(self, *args, dont_rechunk=False, **kwargs):
        if dont_rechunk:
            super().save(*args, **kwargs)
            return
        
        # Skip short document validation for now to help diagnose the issue
        #if len(self.full_text) < 100:
        #    raise ValidationError("The full text of a document must be at least 100 characters long.")
        
        self.full_text_hash = self.hash_fn(self.full_text)

        # TEMPORARY FIX: Skip duplicate check to allow documents through
        # if Document.filter(collection=self.collection, full_text_hash=self.full_text_hash):
        #    raise DuplicateDocumentError(f"Document with title `{self.title}` has the same contents as another document in the same collection.")

        is_new = (not (d := Document.get_by_id(doc_id=self.id))) or (self.full_text_hash != d.full_text_hash)
        super().save(*args, **kwargs)
        
        if is_new:
            self.ingestion_complete = False
            result = None
            try:
                result = create_chunks.delay(str(self.id)) # type: ignore
                for _ in range(4):
                    if state(result.status) == state(FAILURE):
                        raise Exception(f"Task failed")
                    if state(result.status) in [state(RECEIVED), state(STARTED), state(SUCCESS)]:
                        return
                    time.sleep(1)
                raise Exception("Task was not received in time")
            except Exception as e:
                logger.error(f"Error creating chunks for document {self.id}: {str(e)}")
                if result:
                    result.revoke()
                # TEMPORARY FIX: Don't delete documents on error
                # self.delete()
                
                # Instead, mark the document as complete but with error
                self.ingestion_complete = True
                super().save(dont_rechunk=True)

    def move_to(self, new_collection):
        """Move this document to a new collection"""
        if not new_collection.user_can_edit(self.ingested_by):
            raise ValidationError("User does not have permission to move documents to this collection")
        self.collection = new_collection
        self.save()

    def delete(self, *args, **kwargs):
        TextChunk.objects.filter(doc_id=self.id).delete()
        return super().delete(*args, **kwargs)


    

    def __str__(self):
        return f'{ContentType.objects.get_for_model(self)} -- {self.title} in {self.collection.name}'

    @property
    def original_text(self):
        return self.full_text


class VTTDocument(Document):
    audio_file = models.FileField(upload_to='stt_audio/',
                                null=True,
                                validators=[FileExtensionValidator(['mp4',
                                                                    'ogg',
                                                                    'opus',
                                                                    'm4a',
                                                                    'aac'
                                                                    ])])


    


    # def save(self, *args, **kwargs):
    #     self.extract_text()
    #     super().save(*args, **kwargs)

    # def create_chunks(self):
    #     chunk_size = apps.get_app_config('aquillm').chunk_size
    #     overlap = apps.get_app_config('aquillm').chunk_overlap
    #     chunk_pitch = chunk_size - overlap
        
    #     segments = self.transcription['segments']

    #     length = 0
    #     for segment in segments:
    #         segment['start_char'] = length
    #         length += len(segment['text'])
    #         segment['end_char'] = length - 1
        
    #     def get_segment_index_by_offset(offset): # gets the segment containing the offset
    #         for idx, segment in enumerate(segments):
    #             if (segment['start_char'] <= offset and segment['end_char'] >= offset) or segment is segments[-1]:
    #                 return idx
        
    #     content_type = ContentType.objects.get_for_model(self)
    #     TextChunk.objects.filter(content_type=content_type, object_id=self.id).delete()
    #     last_character = len(self.full_text) - 1

    #     chunks = list([TextChunk(
    #         content = '\n'.join([segment['text'] for segment in segments[get_segment_index_by_offset(chunk_pitch * i) : get_segment_index_by_offset(chunk_pitch * i + chunk_size)]]),
    #         start_position = segments[get_segment_index_by_offset(chunk_pitch * i)]['start_char'],
    #         end_position = segments[get_segment_index_by_offset(chunk_pitch * i + chunk_size)]['end_char'],
    #         start_time = segments[get_segment_index_by_offset(chunk_pitch * i)]['start'],
    #         content_type = content_type,
    #         object_id = self.id,
    #         chunk_number = i) for i in range(last_character // chunk_pitch + 1)
    #     ])
    #     for chunk in chunks:
    #         chunk.save()
                


# TODO: figure out how to get rid of this without breaking migrations
def validate_pdf_extension(value):
    if not value.name.endswith('.pdf'):
        raise ValidationError('File must be a PDF')
    


#Currently Working On
class HandwrittenNotesDocument(Document):
    image_file = models.ImageField(
        upload_to='handwritten_notes/', 
        validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])],
        help_text="Upload an image of handwritten notes"
    )
    
    convert_to_latex = False  
    bypass_extraction = False  
    bypass_min_length = True 
    
    def __init__(self, *args, **kwargs):
        self.convert_to_latex = kwargs.pop('convert_to_latex', False) if 'convert_to_latex' in kwargs else False
        self.bypass_extraction = kwargs.pop('bypass_extraction', False) if 'bypass_extraction' in kwargs else False
        
        super().__init__(*args, **kwargs)

    def save(self, *args, **kwargs):
        if not self.pk and not self.bypass_extraction:  # Only extract text on first save, unless bypassed
            self.extract_text()
            self.full_text_hash = hashlib.sha256(self.full_text.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)

    def extract_text(self):
        try:
            # Process directly with the file object or storage file
            if default_storage.exists(self.image_file.name):
                with default_storage.open(self.image_file.name, 'rb') as image_file:
                    # Process directly with the file object
                    result = extract_text_from_image(image_file, convert_to_latex=self.convert_to_latex)
            elif hasattr(self.image_file, 'read'):
                if hasattr(self.image_file, 'seek'):
                    self.image_file.seek(0)
                
                # Process directly with the file object
                result = extract_text_from_image(self.image_file, convert_to_latex=self.convert_to_latex)
                
                if hasattr(self.image_file, 'seek'):
                    self.image_file.seek(0)
            else:
                raise FileNotFoundError(f"Cannot access image file: {self.image_file.name}")
                
            # Process the result
            self.full_text = result.get('extracted_text', '')
            
            if self.convert_to_latex and 'latex_text' in result:
                latex = result.get('latex_text', '')
                if latex and latex != "NO MATH CONTENT":
                    self.full_text += "\n\n==== LATEX VERSION ====\n\n" + latex
            
            if not self.full_text or self.full_text == "NO READABLE TEXT":
                self.full_text = "No readable text could be extracted from this image."
                
        except Exception as e:
            self.full_text = f"Image text extraction failed. Please try again."
            raise
            
    @property
    def latex_content(self):
        if "==== LATEX VERSION ====" in self.full_text:
            parts = self.full_text.split("==== LATEX VERSION ====", 1)
            if len(parts) > 1:
                latex_text = parts[1].strip()
                latex_text = latex_text.replace("==== LATEX VERSION ====", "")
                return latex_text
        return ""
            
    @property
    def has_latex(self):
        return "==== LATEX VERSION ====" in self.full_text
        
    @property
    def original_text(self):
        if "==== LATEX VERSION ====" in self.full_text:
            return self.full_text.split("==== LATEX VERSION ====", 1)[0].strip()
        return self.full_text
    

class PDFDocument(Document):
    pdf_file = models.FileField(upload_to= 'pdfs/', max_length=500, validators=[FileExtensionValidator(['pdf'])])
    zotero_item_key = models.CharField(max_length=100, null=True, blank=True, db_index=True, help_text="Zotero item key to prevent duplicate syncing")

    def save(self, *args, dont_rechunk=False, **kwargs):

        if not dont_rechunk:
            self.extract_text()
        super().save(*args, dont_rechunk=dont_rechunk, **kwargs)

    def extract_text(self):
        text = ""
       
        reader = PdfReader(self.pdf_file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
        self.full_text = text.replace('\0', '')


class TeXDocument(Document):
    pdf_file = models.FileField(upload_to= 'pdfs/', null=True)

    pass

class RawTextDocument(Document):
    source_url = models.URLField(max_length=2000, null=True, blank=True)
    pass

DESCENDED_FROM_DOCUMENT = [
    PDFDocument,
    TeXDocument,
    RawTextDocument,
    VTTDocument,
    HandwrittenNotesDocument,
]

DocumentChild = PDFDocument | TeXDocument | RawTextDocument | VTTDocument | HandwrittenNotesDocument

class TextChunkQuerySet(models.QuerySet):
    def filter_by_documents(self, docs):
        ids = [doc.id for doc in docs]
        return self.filter(doc_id__in=ids)

def doc_id_validator(id):
    if sum([t.objects.filter(id=id).exists() for t in DESCENDED_FROM_DOCUMENT]) != 1:
        raise ValidationError("Invalid Document UUID -- either no such document or multiple")
    

class TextChunk(models.Model):
    content = models.TextField()
    start_position = models.PositiveIntegerField()
    end_position = models.PositiveIntegerField()

    start_time = models.FloatField(null=True)
    chunk_number = models.PositiveIntegerField()
    embedding = VectorField(dimensions=1024, blank=True, null=True)

    
    doc_id = models.UUIDField(editable=False,
                                validators=[doc_id_validator])
    

    @property
    def document(self) -> DocumentChild:
        ret = None
        for t in DESCENDED_FROM_DOCUMENT:
            doc = t.objects.filter(id=self.doc_id).first()
            if doc:
                ret = doc
        if not ret:
            raise ValidationError(f"TextChunk {self.pk} is not associated with a document!")
        return ret

    @document.setter
    def document(self, doc):
        self.doc_id = doc.id


    objects = TextChunkQuerySet.as_manager()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['doc_id', 'start_position', 'end_position'],
                name='unique_chunk_position_per_document'
            ),
            models.UniqueConstraint(
                fields=['doc_id', 'chunk_number'],
                name='uniqe_chunk_per_document'
            )
        ]
        indexes = [
            models.Index(fields=['doc_id', 'start_position', 'end_position']),
            HnswIndex(
                name='chunk_embedding_index',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            ),
        ]
        ordering = ['doc_id', 'chunk_number']

    def save(self, *args, **kwargs):
        if self.start_position >= self.end_position:
            raise ValueError("end_position must be greater than start_position")
        if not self.embedding:
            self.get_chunk_embedding()

        super().save(*args, **kwargs)

    @retry(wait=wait_exponential())
    def get_chunk_embedding(self, callback:Optional[Callable[[], None]]=None):
        self.embedding = get_embedding(self.content, input_type='search_document')
        if callback:
            callback()
    @classmethod
    def rerank(cls, query:str, chunks, top_k: int):
        cohere = apps.get_app_config('aquillm').cohere_client # type: ignore
        response = cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=list([{"content": chunk.content, "id": chunk.pk} for chunk in chunks]),
            rank_fields=['content'],
            top_n=top_k,
            return_documents=True 
        )
        ranked_list = list([result.document.id for result in response.results])
        preserved = Case(*[When(pk=pk, then=pos) for pos, pk in enumerate(ranked_list)])
        return cls.objects.filter(pk__in=ranked_list).order_by(preserved)


    @classmethod
    def text_chunk_search(cls, query:str, top_k: int, docs: List[DocumentChild]):
        vector_top_k = apps.get_app_config('aquillm').vector_top_k # type: ignore
        trigram_top_k = apps.get_app_config('aquillm').trigram_top_k # type: ignore

        try:
            vector_results = cls.objects.filter_by_documents(docs).order_by(L2Distance('embedding', get_embedding(query)))[:vector_top_k] # type: ignore
            trigram_results = cls.objects.filter_by_documents(docs).annotate(similarity = TrigramSimilarity('content', query) # type: ignore
            ).filter(similarity__gt=0.000001).order_by('-similarity')[:trigram_top_k]
            reranked_results = cls.rerank(query, vector_results | trigram_results, top_k)
            return vector_results, trigram_results, reranked_results
        except DatabaseError as e:
            logger.error(f"Database error during search: {str(e)}")
            raise e
        except ValidationError as e:
            logger.error(f"Validation error during search: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            raise e


# Pulls the default system prompt from the app config (set in apps.py)
def get_default_system_prompt():
    return apps.get_app_config('aquillm').system_prompt


class WSConversation(models.Model):
    owner = models.ForeignKey(User, related_name='ws_conversations', on_delete=models.CASCADE)
    # System prompt for this conversation — previously stored inside the JSON blob,
    # now its own field so it can be read/updated without touching the messages
    system_prompt = models.TextField(default=get_default_system_prompt, blank=True)
    name = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        if not self.pk:
            self.created_at = timezone.now()
        self.updated_at = timezone.now()
        return super().save(*args, **kwargs)
    
        
    def set_name(self):
        import asyncio
        from asgiref.sync import async_to_sync

        system_prompt="""
        This is a conversation between a large langauge model and a user.
        Come up with a brief, roughly 3 to 10 word title for the conversation capturing what the user asked.
        Respond only with the title.
        As an example, if the conversation begins 'What is apple pie made of?', your response should be 'Apple Pie Ingredients'.
        The title should capture what is being asked, not what the assistant responded with.
        If there is not enough information to name the conversation, simply return 'Conversation'.
        """

        # Use the configured LLM interface instead of hardcoded Anthropic client
        llm_interface = apps.get_app_config('aquillm').llm_interface # type: ignore
        # Query the first 2 messages from the Message table (instead of parsing JSON blob)
        first_two = list(self.db_messages.order_by('sequence_number')[:2].values('role', 'content'))
        first_two_messages = str(first_two)

        # Build kwargs compatible with the LLM interface
        llm_args = {
            **llm_interface.base_args,  # Include base args (model, etc.)
            'max_tokens': 30,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': first_two_messages}]
        }

        # Run the async method synchronously
        @async_to_sync
        async def get_title():
            response = await llm_interface.get_message(**llm_args)
            return response.text

        title_text = get_title()
        self.name = title_text if title_text else 'Conversation'
        self.save()


# Stores individual messages — replaces the old JSON blob approach where all messages
# were stored in a single column on WSConversation. Each message is now its own row,
# making them individually queryable (e.g. "find all 5-star messages across all conversations").
#
# Uses a "wide table" design: all 3 message types (user, assistant, tool) share one table.
# Role-specific fields are nullable — a user message won't have 'model' or 'tool_name',
# and those columns will just be NULL for that row.
class Message(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant'), ('tool', 'Tool')]
    FOR_WHOM_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]

    # Core fields (used by all message types)
    conversation = models.ForeignKey(WSConversation, on_delete=models.CASCADE, related_name='db_messages')  # FK back to the conversation this message belongs to
    message_uuid = models.UUIDField(default=uuid.uuid4, db_index=True)  # unique ID sent to frontend (not guessable like sequential IDs)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)  # 'user', 'assistant', or 'tool'
    content = models.TextField()  # the actual message text
    rating = models.PositiveSmallIntegerField(null=True, blank=True)  # user rating 1-5, null if unrated
    feedback_text = models.TextField(null=True, blank=True)  # optional user feedback text
    sequence_number = models.PositiveIntegerField()  # position in conversation (0, 1, 2, ...) — determines display order
    created_at = models.DateTimeField(auto_now_add=True)  # when this message was created

    # AssistantMessage-specific fields (NULL for user and tool messages)
    model = models.CharField(max_length=100, null=True, blank=True)  # which LLM generated this (e.g. 'claude-3-7-sonnet-latest')
    stop_reason = models.CharField(max_length=50, null=True, blank=True)  # why the LLM stopped ('end_turn' or 'tool_use')
    tool_call_id = models.CharField(max_length=100, null=True, blank=True)  # ID of tool call if the LLM invoked a tool
    tool_call_name = models.CharField(max_length=100, null=True, blank=True)  # name of tool called (e.g. 'vector_search')
    tool_call_input = models.JSONField(null=True, blank=True)  # arguments the LLM passed to the tool
    usage = models.PositiveIntegerField(default=0)  # token count for this response

    # ToolMessage-specific fields (NULL for user and assistant messages)
    tool_name = models.CharField(max_length=100, null=True, blank=True)  # which tool produced this result
    arguments = models.JSONField(null=True, blank=True)  # arguments the tool was called with
    for_whom = models.CharField(max_length=10, choices=FOR_WHOM_CHOICES, null=True, blank=True)  # who gets the result ('assistant' or 'user')
    result_dict = models.JSONField(null=True, blank=True)  # the tool's output data

    class Meta:
        ordering = ['conversation', 'sequence_number']  # default ordering: by conversation, then by position
        indexes = [models.Index(fields=['rating'])]  # index on rating for fast queries like "find all 5-star messages"


class ConversationFile(models.Model):
    file = models.FileField(upload_to='conversation_files/')
    name = models.CharField(max_length=200)
    conversation = models.ForeignKey(WSConversation, on_delete=models.CASCADE, related_name='files')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    message_uuid = models.UUIDField(null=True, blank=True)
    def __str__(self):
        return f"File {self.file.name} for conversation {self.conversation.id}"


class EmailWhitelist(models.Model):
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.email


class GeminiAPIUsage(models.Model):
    """Model to track Gemini API usage and costs"""
    timestamp = models.DateTimeField(auto_now_add=True)
    operation_type = models.CharField(max_length=100, help_text="Type of operation (e.g., 'OCR', 'Handwritten Notes')")
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)

    # Constants for pricing (can be updated as needed)
    INPUT_COST_PER_1K = 0.0005  # $0.0005 per 1,000 input tokens
    OUTPUT_COST_PER_1K = 0.0015  # $0.0015 per 1,000 output tokens

    class Meta:
        verbose_name = "Gemini API Usage"
        verbose_name_plural = "Gemini API Usage"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.operation_type} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @classmethod
    def calculate_cost(cls, input_tokens, output_tokens):
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1000) * cls.INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * cls.OUTPUT_COST_PER_1K
        return input_cost + output_cost

    @classmethod
    def log_usage(cls, operation_type, input_tokens, output_tokens):
        """Log API usage and return the cost"""
        cost = cls.calculate_cost(input_tokens, output_tokens)
        usage = cls.objects.create(
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        return usage

    @classmethod
    def get_total_stats(cls):
        """Get aggregated usage statistics"""
        from django.db.models import Sum, Count

        stats = cls.objects.aggregate(
            total_input_tokens=Sum('input_tokens'),
            total_output_tokens=Sum('output_tokens'),
            total_cost=Sum('cost'),
            api_calls=Count('id')
        )
        return stats