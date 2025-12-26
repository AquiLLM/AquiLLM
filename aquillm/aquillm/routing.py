from django.urls import re_path
from . import crawl_status_consumers, zotero_sync_consumers

websocket_urlpatterns = [
    # Route for crawl status updates, requires authentication
    re_path(r'ws/crawl_status/$', crawl_status_consumers.CrawlStatusConsumer.as_asgi()),
    # Route for Zotero sync status updates
    re_path(r'ws/zotero_sync/$', zotero_sync_consumers.ZoteroSyncConsumer.as_asgi()),
]