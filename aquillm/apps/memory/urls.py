from django.urls import path
from . import views

urlpatterns = [
    path('profile-memory/', views.profile_memory_list, name='profile-memory-list'),
    path('profile-memory/<int:fact_id>/', views.profile_memory_detail, name='profile-memory-detail'),
]
