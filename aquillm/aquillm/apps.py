from django.apps import AppConfig

from django.template import Engine, Context
import cohere
import openai
import anthropic
import google.generativeai as genai
from os import getenv
from typing import TypedDict


from .llm import LLMInterface, ClaudeInterface, OpenAIInterface
from .settings import DEBUG
RAG_PROMPT_STRING = """
<context>
RAG Search Results:

{% for chunk in message.context_chunks.all %}
    [{{ forloop.counter }}] {{ chunk.document.title }} chunk #{{chunk.chunk_number}}

    {{ chunk.content }}

{% endfor %}
</context>
<user-query>
    {{ message.content }}
</user-query>
"""




def get_embedding_func(cohere_client):


    def get_embedding(query: str, input_type: str='search_query'):
        if input_type not in ('search_document', 'search_query', 'classification', 'clustering'):
            raise ValueError(f'bad input type to embedding call: {input_type}')
        response = cohere_client.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type=input_type
        )
        return response.embeddings[0]
    return get_embedding

class AquillmConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'aquillm'
    cohere_client = None
    openai_client = None
    anthropic_client = None
    async_anthropic_client = None
    get_embedding = None
    llm_interface: LLMInterface = None
    system_prompt = "You are a helpful assistant embedded in a retrieval augmented generation system."

    google_genai_client = None
    default_llm = "CLAUDE"
    
    
    vector_top_k = 30
    trigram_top_k = 30
    rag_prompt_template = Engine().from_string(RAG_PROMPT_STRING)



    chunk_size = 2048
    chunk_overlap = 512 # at each end.
#   |-----------CHUNK-----------|
#   <---------chunk_size-------->
#                       <------->  chunk_overlap
#                       |-----------CHUNK-----------|
    def ready(self):

        self.cohere_client = cohere.Client(getenv('COHERE_KEY'))
        self.openai_client = openai.AsyncOpenAI()
        self.anthropic_client = anthropic.Anthropic()
        self.async_anthropic_client = anthropic.AsyncAnthropic()
        self.async_anthropic_bedrock_client = anthropic.AsyncAnthropicBedrock(
            aws_region='us-east-1',
            aws_access_key=getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_key=getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.get_embedding = get_embedding_func(self.cohere_client)
        llm_choice = getenv('LLM_CHOICE', self.default_llm)
        if llm_choice == 'CLAUDE':
            self.llm_interface = ClaudeInterface(self.async_anthropic_client)
        elif llm_choice == 'OPENAI':
            self.llm_interface = OpenAIInterface(self.openai_client, "gpt-4o")
        elif llm_choice == 'BEDROCK-CLAUDE':
            self.llm_interface = ClaudeInterface(self.async_anthropic_bedrock_client, model_override='arn:aws:bedrock:us-east-1:744423739991:inference-profile/us.anthropic.claude-opus-4-1-20250805-v1:0')
        else:
            raise ValueError(f"Invalid LLM choice: {llm_choice}")

