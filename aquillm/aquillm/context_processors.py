from .models import WSConversation
from django.urls import get_resolver
from .models import UserSettings

def nav_links(request):
    return {
        'nav_links': [
            {"url": "new_ws_convo", "text": "New Conversation"},
            {"url": "user_ws_convos", "text": "Old Conversations"},
            {"url": "search", "text": "Search"},
            {"url": "user_collections", "text": "Collections"},
            # {"url": "insert_arxiv", "text": "Ingest from arXiv"},
            # {"url": "ingest_pdf", "text": "Ingest PDF"},
            # {"url": "ingest_vtt", "text": "Ingest Transcript"},
            # {"url": "ingest_handwritten_notes", "text": "Ingest Handwritten Notes"},
            # {"url": "gemini_cost_monitor", "text": "Gemini Costs"}

        ]
    }
resolver=get_resolver()
_api_url_dict = {key: "/" + url_pattern[0][0][0] for key, url_pattern in resolver.reverse_dict.items() if url_pattern[0][0][0].split('/')[0] == 'api' and isinstance(key, str)} # type: ignore
_page_url_dict = {key: "/" + url_pattern[0][0][0] for key, url_pattern in resolver.reverse_dict.items() if url_pattern[0][0][0].split('/')[0] != 'api' and isinstance(key, str)} # type: ignore

def api_urls(request):
    # Add the handwritten notes URL to the API URLs
    custom_urls = _api_url_dict.copy()
    custom_urls['api_ingest_handwritten_notes'] = '/aquillm/ingest_handwritten_notes/'
    return {'api_urls': custom_urls}

def page_urls(request):
    return {'page_urls': _page_url_dict}

def user_conversations(request):
    if request.user.is_authenticated:
        convos = WSConversation.objects.filter(owner=request.user).order_by('-updated_at')
        return {'conversations': convos}
    return {}

def theme_settings(request):
    if request.user.is_authenticated:
        try:
            settings = UserSettings.objects.get(user=request.user)
        except UserSettings.DoesNotExist:
            settings = None
    else:
        settings = None
    return {'user_theme_settings': settings}