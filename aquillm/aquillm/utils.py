
import logging
logger = logging.getLogger(__name__)




from django.apps import apps


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
