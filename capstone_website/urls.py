"""
Main URL configuration for capstone website
"""
from django.urls import path, include

urlpatterns = [
    path('', include('main_app.urls')),
]
