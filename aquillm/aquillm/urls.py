"""
URL configuration for aquillm project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from debug_toolbar.toolbar import debug_toolbar_urls

from chat import views as chat_views
from chat.views import urlpatterns as chat_urlpatterns
from . import views, api_views
from .views import urlpatterns as page_urlpatterns
from .settings import DEBUG
from .api_views import urlpatterns as api_urlpatterns
from .views import UserSettingsPageView
from . import zotero_views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/", include(api_urlpatterns)),
    path("aquillm/", include(page_urlpatterns)),
    path("chat/", include(chat_urlpatterns)),
    path('accounts/', include('allauth.urls')),

    path("admin/", admin.site.urls),
    path("new_ws_convo/", chat_views.new_ws_convo, name="new_ws_convo"),

    path("health/", views.health_check, name="health"),
    path("ready/", views.health_check, name="ready"),
    path("health", views.health_check),
    path("ready", views.health_check),

    path('user-settings/', UserSettingsPageView.as_view(), name='user-settings-page'),

    # Zotero integration URLs
    path('zotero/settings/', zotero_views.zotero_settings, name='zotero_settings'),
    path('zotero/connect/', zotero_views.zotero_connect, name='zotero_connect'),
    path('zotero/callback/', zotero_views.zotero_callback, name='zotero_callback'),
    path('zotero/disconnect/', zotero_views.zotero_disconnect, name='zotero_disconnect'),
    path('zotero/sync/', zotero_views.zotero_sync, name='zotero_sync'),
    path('zotero/sync/status/', zotero_views.zotero_sync_status, name='zotero_sync_status'),
] + debug_toolbar_urls()

if DEBUG:
   urlpatterns += [
       path("debug_models/", views.debug_models, name="debug_models"),
   ]