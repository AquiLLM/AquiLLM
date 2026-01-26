from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('problem-statement/', views.problem_statement, name='problem_statement'),
]
