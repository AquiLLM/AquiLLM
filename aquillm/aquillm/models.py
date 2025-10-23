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

from .llm import Conversation as convo_model

logger = logging.getLogger(__name__)

from pydantic_core import to_jsonable_python

from .celery import app
from celery.states import state, RECEIVED, STARTED, SUCCESS, FAILURE

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

channel_layer = get_channel_layer()
assert (channel_layer is not None and
    hasattr(channel_layer, 'group_send')) # keeps type checker happy

type DocumentChild = PDFDocument | TeXDocument | RawTextDocument | VTTDocument


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
        doc.delete()
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
        if len(self.full_text) < 100:
            raise ValidationError("The full text of a document must be at least 100 characters long.")
        self.full_text_hash = self.hash_fn(self.full_text)
        
        # if Document.filter(collection=self.collection, full_text_hash=self.full_text_hash):
        #     raise DuplicateDocumentError(f"Document with title `{self.title}` has the same contents as another document in the same collection.")
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
                self.delete()

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
    
    

class PDFDocument(Document):
    pdf_file = models.FileField(upload_to= 'pdfs/', max_length=500, validators=[FileExtensionValidator(['pdf'])])

    def save(self, dont_rechunk=False, *args, **kwargs):
        
        if not dont_rechunk:
            self.extract_text()
        super().save(*args, **kwargs)

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
    pass


class TextChunkQuerySet(models.QuerySet):
    def filter_by_documents(self, docs):
        ids = [doc.id for doc in docs]
        return self.filter(doc_id__in=ids)


DESCENDED_FROM_DOCUMENT = [PDFDocument, TeXDocument, RawTextDocument, VTTDocument]

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


class WSConversation(models.Model):
    owner = models.ForeignKey(User, related_name='ws_conversations', on_delete=models.CASCADE)
    convo = models.JSONField(blank=True, null=True, default=convo_model.get_empty_conversation)
    name = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(editable=False)
    updated_at = models.DateTimeField()

    @property
    def convo_object(self) -> convo_model:
        t = type(self.convo)
        if t == dict:
            return convo_model.model_validate(self.convo)
        elif t == str:
            return convo_model.model_validate_json(self.convo)
        else: 
            raise ValueError(f"Invalid type for convo from database: {t}")

    @convo_object.setter
    def convo_object(self, convo: convo_model):
        self.convo = to_jsonable_python(convo)

    def save(self, *args, **kwargs):
        if not self.pk:
            self.created_at = timezone.now()
        self.updated_at = timezone.now()
        return super().save(*args, **kwargs)
    
        
    def set_name(self):
        system_prompt="""
        This is a conversation between a large langauge model and a user.
        Come up with a brief, roughly 3 to 10 word title for the conversation capturing what the user asked.
        Respond only with the title. 
        As an example, if the conversation begins 'What is apple pie made of?', your response should be 'Apple Pie Ingredients'.
        The title should capture what is being asked, not what the assistant responded with.
        If there is not enough information to name the conversation, simply return 'Conversation'.
        """
        anthropic_client = apps.get_app_config('aquillm').anthropic_client # type: ignore
        first_two_messages = str(self.convo['messages'][:2])
        claude_args = {'model': 'claude-3-5-sonnet-20240620',
            'max_tokens': 30,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': first_two_messages}]}
        message = anthropic_client.messages.create(**claude_args)
        self.name = message.content[0].text
        self.save()


class EmailWhitelist(models.Model):
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.email