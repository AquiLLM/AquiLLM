
import logging
import sys
logger = logging.getLogger(__name__)




from django.apps import apps


def get_debug_traceback_html():
    """Generate Django's HTML traceback page for the current exception. Returns None if DEBUG is off."""
    from .settings import DEBUG
    if not DEBUG:
        return None
    from django.views.debug import ExceptionReporter
    return ExceptionReporter(None, *sys.exc_info()).get_traceback_html()


def get_embedding(query: str, input_type: str='search_query'):
    cohere_client = apps.get_app_config('aquillm').cohere_client
    if cohere_client is None:
        raise Exception("Cohere client is still none while app is running")
    if input_type not in ('search_document', 'search_query', 'classification', 'clustering'):
        raise ValueError(f'bad input type to embedding call: {input_type}')
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type=input_type
    )
    return response.embeddings[0]
