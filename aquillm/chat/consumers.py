from os import getenv
from typing import Optional
from json import loads, dumps
from uuid import UUID
from base64 import b64decode, b64encode
from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.auth import AuthMiddlewareStack
from channels.db import database_sync_to_async, aclose_old_connections
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from django.apps import apps

from pydantic import ValidationError
import aquillm.llm
from aquillm.llm import UserMessage, Conversation, LLMTool, LLMInterface, test_function, ToolChoice, llm_tool, ToolResultDict, message_to_user
from aquillm.settings import DEBUG

from aquillm.models import ConversationFile, TextChunk, Collection, CollectionPermission, WSConversation, Document, DocumentChild
# Adapter functions that handle converting between Pydantic (runtime) and Django (database) message formats
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db, build_frontend_conversation_json

from anthropic._exceptions import OverloadedError

import logging
logger = logging.getLogger(__name__)

import io
# necessary so that when collections are set inside the consumer, it changes inside the vector_search closure as well. 
class CollectionsRef:
    def __init__(self, collections: list[int]):
        self.collections = collections

class ChatRef:
    def __init__(self, chat: 'ChatConsumer'):
        self.chat = chat

def get_vector_search_func(user: User, col_ref: CollectionsRef): 
    @llm_tool(
        param_descs={"search_string": "The string to search by. Often it helps to phrase it as a question. ",
                     "top_k": "The number of results to return. Start low and increase if the desired information is not found. Go no higher than about 15."},
        required=['search_string', 'top_k'],
        for_whom='assistant'

    )
    def vector_search(search_string: str, top_k: int) -> ToolResultDict:
        """
        Uses a combination of vector search, trigram search and reranking to search the documents available to the user.
        """
        docs = Collection.get_user_accessible_documents(user, Collection.objects.filter(id__in=col_ref.collections))
        if not docs:
            return {"exception": "No documents to search! Either no collections were selected, or the selected collections are empty."}
        _,_,results = TextChunk.text_chunk_search(search_string, top_k, docs)
        ret = {"result": {f"[Result {i+1}] -- {chunk.document.title} chunk #: {chunk.chunk_number} chunk_id:{chunk.id}": chunk.content for i, chunk in enumerate(results)}}
        return ret
    
    return vector_search


def get_document_ids_func(user: User, col_ref: CollectionsRef) -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=[],
        param_descs={}
    )
    def document_ids() -> ToolResultDict:
        """
        Get the names and IDs of all documents in the selected collections. When a user asks to see a document in full, or to search a single document, use this to get its ID.
        """
        docs = Collection.get_user_accessible_documents(user, Collection.objects.filter(id__in=col_ref.collections))
        if not docs:
            return {"exception": "No documents to search! Either no collections were selected, or the selected collections are empty."}
        return {"result": {doc.title: str(doc.id) for doc in docs}}
    return document_ids

def get_whole_document_func(user: User, chat_ref: ChatRef) -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=['doc_id'],
        param_descs={'doc_id': 'UUID (as as string) of the document to return in full'}
    )
    def whole_document(doc_id: str) -> ToolResultDict:
        """
        Get the full text of a document. Use when a user asks you to get a full document. Depending on the size of the document, this will not always be possible. 
        """
        try:
            doc_uuid = UUID(doc_id)
        except Exception as e:
            return {"exception": f"Invalid document ID: {doc_id}"}
        doc: Optional[DocumentChild] = Document.get_by_id(doc_uuid)
        if doc is None:
            return {"exception": f"Document {doc_id} does not exist!"}
        if not doc.collection.user_can_view(user):
            return {"exception": f"User cannot access document {doc_id}!"}
        token_count = async_to_sync(chat_ref.chat.llm_if.token_count)(chat_ref.chat.convo, doc.full_text)
        if token_count > 150000:
            return {"exception": f"Document {doc_id} is too large to open in this chat."}
        return {"result": doc.full_text}
    
    return whole_document

def get_search_single_document_func(user: User) -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=['doc_id', 'query'],
        param_descs={'doc_id': 'UUID (as a string) of the document to search.',
                     'search_string': 'String to search the contents of the document by.',
                     'top_k': 'Number of search results to return.'}
    )
    def search_single_document(doc_id: str, search_string: str, top_k: int) -> ToolResultDict:
        """
        Use vector search to search the text of a single document.
        """
        doc_uuid = UUID(doc_id)
        doc = Document.get_by_id(doc_uuid)
        if doc is None:
            return {"exception": f"Document {doc_id} does not exist!"}
        if not doc.collection.user_can_view(user):
            return {"exception": f"User cannot access document {doc_id}!"}
        _,_,results = TextChunk.text_chunk_search(search_string, top_k, [doc])
        ret = {"result": {f"[Result {i+1}] -- {chunk.document.title} chunk #: {chunk.chunk_number} chunk_id:{chunk.id}": chunk.content for i, chunk in enumerate(results)}}
        return ret
    
    return search_single_document
    
    return search_single_document
def get_more_context_func(user: User) -> LLMTool:
    @llm_tool(
            for_whom='assistant',
            required=['adjacent_chunks', 'chunk_id'],

            param_descs={'chunk_id': 'ID number of the chunk for which more context is desired',
                         'adjacent_chunks': 'How many chunks on either side to return. Start small and work up, if you think expanding the context will provide more useful info. Go no higher than 10.'},
    )
    def more_context(chunk_id: int, adjacent_chunks: int) -> ToolResultDict:
        """
        Get adjacent text chunks on either side of a given chunk.
        Use this when a search returned something relevant, but it seemed like the information was cut off.
        """
        if adjacent_chunks < 1 or adjacent_chunks > 10:
            return {"exception": f"Invalid value for adjacent_chunks!"}
        central_chunk = TextChunk.objects.filter(id=chunk_id).first()
        if central_chunk is None:
            return {"exception": f"Text chunk {chunk_id} does not exist!"}
        if not central_chunk.document.collection.user_can_view(user):
            return {"exception": f"User cannot access document containing {chunk_id}!"}
        central_chunk_number = central_chunk.chunk_number
        bottom = central_chunk_number - adjacent_chunks
        top = central_chunk_number + adjacent_chunks
        window = TextChunk.objects.filter(doc_id=central_chunk.doc_id, chunk_number__in=range(bottom, top+1)).order_by('chunk_number')
        text_blob = "".join([chunk.content for chunk in window])
        return {"result": f"chunk_numbers:{window.first().chunk_number} -> {window.last().chunk_number} \n\n {text_blob}"}
    return more_context


def get_sky_subtraction_func(chat_consumer: 'ChatConsumer') -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=['object', 'sky'],
        param_descs={'object_id': 'The file ID of the FITS file containing the object to subtract the sky from',
                     'sky_id': 'The file ID of the FITS file of the sky to subtract from the object'}
    )
    def sky_subtraction(object_id: int, sky_id: int) -> ToolResultDict:
        """
        Subtracts the sky from a FITS image of an object.
        
        Use this when a user asks you to subtract the sky from an object, and provides the files, one of the sky and one of the object.
        Specify the IDs of the files in the parameters.
        """
        from astropy.io import fits as fits
        try:
            convo = chat_consumer.db_convo
            object = ConversationFile.objects.filter(id=object_id).first()
            sky = ConversationFile.objects.filter(id=sky_id).first()
            if object is None or sky is None:
                return {"exception": f"One or more files do not exist!"}
            if object.conversation != convo or sky.conversation != convo:
                return {"exception": f"One or more files do not belong to this conversation!"}
            object_file = object.file
            sky_file = sky.file
            object_data = fits.getdata(object_file.open('rb'))
            sky_data = fits.getdata(sky_file.open('rb'))
            if object_data.shape != sky_data.shape:
                return {"exception": f"Wrong dimensions! The object and sky files must have the same dimensions."}
            result = object_data - sky_data
            result_io = io.BytesIO()
            fits.writeto(result_io, result, overwrite=True)
            result_io.seek(0)
            result_file = ContentFile(result_io.read(), name=f"{object_file.name[:-5]}_sky_subtracted.fits")
            result_conversation_file = ConversationFile(file=result_file, conversation=convo, name=f"{object_file.name[:-5]}_sky_subtracted.fits")
            result_conversation_file.save()
            return {"result": f"Sky subtracted!", 'files': [(result_conversation_file.name, result_conversation_file.id)]}
        except Exception as e:
            return {"exception": f"An error occurred while subtracting the sky: {str(e)}"}
    return sky_subtraction


def get_flat_fielding_func(chat_consumer: 'ChatConsumer') -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=['science', 'flat'],
        param_descs={
            'science_id': 'The file ID of the FITS image to be flat-field corrected',
            'flat_id': 'The file ID of the flat-field FITS image to use for correction'
        }
    )
    def flat_fielding(science_id: int, flat_id: int) -> ToolResultDict:
        """
        Applies flat-field correction to a FITS image.

        Use this when a user provides a science image and a flat-field image to correct for detector sensitivity variations.
        """
        from astropy.io import fits
        try:
            convo = chat_consumer.db_convo
            science = ConversationFile.objects.filter(id=science_id).first()
            flat = ConversationFile.objects.filter(id=flat_id).first()
            if science is None or flat is None:
                return {"exception": f"One or more files do not exist!"}
            if science.conversation != convo or flat.conversation != convo:
                return {"exception": f"One or more files do not belong to this conversation!"}
            science_file = science.file
            flat_file = flat.file
            science_data = fits.getdata(science_file.open('rb'))
            flat_data = fits.getdata(flat_file.open('rb'))
            if science_data.shape != flat_data.shape:
                return {"exception": f"Wrong dimensions! Science and flat-field images must have the same shape."}
            # Avoid division by zero
            if (flat_data == 0).any():
                return {"exception": "Flat field image contains zero values, cannot safely divide."}

            result = science_data / flat_data
            result_io = io.BytesIO()
            fits.writeto(result_io, result, overwrite=True)
            result_io.seek(0)
            result_file = ContentFile(result_io.read(), name=f"{science_file.name[:-5]}_flat_corrected.fits")
            result_conversation_file = ConversationFile(
                file=result_file,
                conversation=convo,
                name=f"{science_file.name[:-5]}_flat_corrected.fits"
            )
            result_conversation_file.save()
            return {"result": "Flat-fielding applied!", "files": [(result_conversation_file.name, result_conversation_file.id)]}
        except Exception as e:
            return {"exception": f"An error occurred during flat-fielding: {str(e)}"}
    return flat_fielding


def get_point_source_detection_func(chat_consumer: 'ChatConsumer') -> LLMTool:
    @llm_tool(
        for_whom='assistant',
        required=['image'],
        param_descs={
            'image_id': 'The file ID of the sky-subtracted and flat-fielded FITS image to run source detection on.'
        }
    )
    def detect_point_sources(image_id: int) -> ToolResultDict:
        """
        Detect point sources in a processed FITS image using DAOStarFinder.
        Apply the method of sigma clipping with sigma=3.0, using DAOStarFinder with fwhm=3.0 and threshold=5*std

        Use this after sky subtraction and flat-fielding to extract point sources from the image.
        """
        from astropy.io import fits
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder
        import pandas as pd

        try:
            convo = chat_consumer.db_convo
            image = ConversationFile.objects.filter(id=image_id).first()
            if image is None:
                return {"exception": f"The file does not exist!"}
            if image.conversation != convo:
                return {"exception": f"The file does not belong to this conversation!"}

            image_file = image.file
            data = fits.getdata(image_file.open('rb'))

            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
            sources = daofind(data - median)

            if sources is None or len(sources) == 0:
                return {"result": "No sources detected."}

            # Save the table to CSV
            df = sources.to_pandas()
            csv_io = io.StringIO()
            df.to_csv(csv_io, index=False)
            csv_file = ContentFile(csv_io.getvalue().encode('utf-8'),
                                   name=f"{image_file.name[:-5]}_sources.csv")
            result_conversation_file = ConversationFile(
                file=csv_file,
                conversation=convo,
                name=csv_file.name
            )
            result_conversation_file.save()

            return {
                "result": f"Detected {len(df)} sources.",
                "files": [(result_conversation_file.name, result_conversation_file.id)]
            }
        except Exception as e:
            return {"exception": f"An error occurred during source detection: {str(e)}"}
    return detect_point_sources




class ChatConsumer(AsyncWebsocketConsumer):
    llm_if: LLMInterface = apps.get_app_config('aquillm').llm_interface
    db_convo: Optional[WSConversation] = None
    convo: Optional[Conversation] = None
    tools: list[LLMTool] = []
    user: Optional[User] = None

    # used for if the chat is in a state where nothing further should happen.
    # disables the receive handler
    dead: bool = False 
    
    
    col_ref = CollectionsRef([])
    
    @database_sync_to_async
    def __save(self):
        assert self.db_convo is not None
        # Converts in-memory Pydantic messages to Django Message rows and saves them to the database
        save_conversation_to_db(self.convo, self.db_convo)
        if len(self.convo) >= 2 and not self.db_convo.name:
            self.db_convo.set_name()

    @database_sync_to_async
    def __get_convo(self, convo_id: int, user: User):
        convo = WSConversation.objects.filter(id=convo_id).first()
        if convo: 
            if convo.owner == user:
                return convo
            else:
                return None
        return convo
        
    @database_sync_to_async
    def __get_all_user_collections(self):
        self.col_ref.collections = [col_perm.collection.id for col_perm in CollectionPermission.objects.filter(user=self.user)]
    

    async def connect(self):
        logger.debug(f"ChatConsumer.connect() called")

        async def send_func(convo: Conversation):
            logger.debug(f"send_func called in connect()")
            self.convo = convo
            await self.__save()
            # Builds JSON from Message table rows and sends it to the frontend over WebSocket
            frontend_json = await database_sync_to_async(build_frontend_conversation_json)(self.db_convo)
            await self.send(text_data=dumps({"conversation": frontend_json}))
            logger.debug(f"send_func completed in connect()")

        async def stream_delta_func(message_uuid: UUID, delta: str):
            await self.send(text_data=dumps({
                "stream": {
                    "message_uuid": str(message_uuid),
                    "delta": delta
                }
            }))

        await self.accept()
        logger.debug(f"WebSocket accepted")
        self.user = self.scope['user']
        assert self.user is not None
        logger.debug(f"User: {self.user}")
        await self.__get_all_user_collections()
        logger.debug(f"Collections loaded: {self.col_ref.collections}")
        self.tools = [
                      get_vector_search_func(self.user, self.col_ref),
                      get_more_context_func(self.user),
                      get_document_ids_func(self.user, self.col_ref),
                      get_whole_document_func(self.user, ChatRef(self)),
                      get_search_single_document_func(self.user),
                      get_sky_subtraction_func(self),
                      get_flat_fielding_func(self),
                      get_point_source_detection_func(self)]
        if getenv('LLM_CHOICE') == 'GEMMA3':
            self.tools.append(message_to_user)
        convo_id = self.scope['url_route']['kwargs']['convo_id']
        logger.debug(f"Convo ID: {convo_id}")
        self.db_convo = await self.__get_convo(convo_id, self.user)
        if self.db_convo is None:
            logger.error(f"Invalid conversation ID: {convo_id}")
            self.dead = True
            await self.send('{"exception": "Invalid chat_id"}')

            return
        try:
            # Loads Message rows from the database and converts them to Pydantic objects for runtime use
            self.convo = await database_sync_to_async(load_conversation_from_db)(self.db_convo)
            self.convo.rebind_tools(self.tools)
            logger.debug(f"About to call llm_if.spin() in connect()")
            await self.llm_if.spin(
                self.convo,
                max_func_calls=5,
                max_tokens=2048,
                send_func=send_func,
                stream_delta_func=stream_delta_func
            )
            logger.debug(f"llm_if.spin() completed in connect()")
            return
        except OverloadedError as e:
            logger.error(f"LLM overloaded: {e}")
            self.dead = True
            await self.send('{"exception": "LLM provider is currently overloaded. Try again later."}')
            return
        except Exception as e:
            logger.error(f"Exception in connect(): {e}", exc_info=True)
            if DEBUG:
                raise e
            else:

                await self.send(text_data='{"exception": "A server error has occurred. Try reloading the page"}')

                return



    async def receive(self, text_data):
        logger.debug(f"ChatConsumer.receive() called with data: {text_data[:100]}...")

        @database_sync_to_async
        def _save_files(files: list[ConversationFile]) -> list[ConversationFile]:
            for file in files:
                file.save()
            return files

        async def send_func(convo: Conversation):
            logger.debug("send_func called in receive()")
            await aclose_old_connections()
            self.convo = convo
            await self.__save()
            # Builds JSON from Message table rows and sends it to the frontend over WebSocket
            frontend_json = await database_sync_to_async(build_frontend_conversation_json)(self.db_convo)
            await self.send(text_data=dumps({"conversation": frontend_json}))
            logger.debug("send_func completed in receive()")

        async def stream_delta_func(message_uuid: UUID, delta: str):
            await self.send(text_data=dumps({
                "stream": {
                    "message_uuid": str(message_uuid),
                    "delta": delta
                }
            }))

        async def append(data: dict):
            logger.debug(f"append() called with collections: {data.get('collections', [])}")

            assert self.convo is not None

            self.col_ref.collections = data['collections']
            self.convo += UserMessage.model_validate(data['message'])
            if 'files' in data:
                files = [ConversationFile(
                            file=ContentFile(b64decode(file['base64']),
                                             name=file['filename']),
                            conversation=self.db_convo, name=file['filename'][-200:],
                            message_uuid=self.convo[-1].message_uuid) for file in data['files']]
                await _save_files(files)
            self.convo[-1].tools = self.tools
            self.convo[-1].files = [(file.name, file.id) for file in files]
            self.convo[-1].tool_choice = ToolChoice(type='auto')
            await self.__save()
            logger.debug("append() completed, message added")

        async def rate(data: dict):
            assert self.convo is not None
            uuid_str = data['uuid']
            rating = data['rating']

            # Update just the single Message row in the database (no need to delete + recreate all messages)
            await database_sync_to_async(
                lambda: self.db_convo.db_messages.filter(message_uuid=uuid_str).update(rating=rating)
            )()

            # Also update the in-memory Pydantic model so the rating stays correct
            # if the conversation is saved again later during this session
            for msg in self.convo:
                if str(msg.message_uuid) == uuid_str:
                    msg.rating = rating
                    break

        async def feedback(data: dict):
            assert self.convo is not None
            uuid_str = data['uuid']
            feedback_text = data['feedback_text']

            # Update just the single Message row in the database
            await database_sync_to_async(
                lambda: self.db_convo.db_messages.filter(message_uuid=uuid_str).update(feedback_text=feedback_text)
            )()

            # Also update the in-memory Pydantic model
            for msg in self.convo:
                if str(msg.message_uuid) == uuid_str:
                    msg.feedback_text = feedback_text
                    break

        if not self.dead:
            try:
                data = loads(text_data)
                action = data.pop('action', None)
                logger.debug(f"Action: {action}")
                if action == 'append':
                    await append(data)
                elif action == 'rate':
                    await rate(data)
                elif action == 'feedback':
                    await feedback(data)
                else:
                    raise ValueError(f'Invalid action "{action}"')
                logger.debug("About to call llm_if.spin() in receive()")
                await self.llm_if.spin(
                    self.convo,
                    max_func_calls=5,
                    max_tokens=2048,
                    send_func=send_func,
                    stream_delta_func=stream_delta_func
                )
                logger.debug("llm_if.spin() completed in receive()")
            except Exception as e:
                logger.error(f"Exception in receive(): {e}", exc_info=True)
                if DEBUG:
                    raise e
                else:
                    await self.send(text_data='{"exception": "A server error has occurred. Try reloading the page"}')

