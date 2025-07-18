from django.urls import re_path
from . import crawl_status_consumers

websocket_urlpatterns = [
    # Route for crawl status updates, requires authentication
    re_path(r'ws/crawl_status/$', crawl_status_consumers.CrawlStatusConsumer.as_asgi()),
]