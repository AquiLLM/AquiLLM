from django.shortcuts import render


def index(request):
    """Main landing page view"""
    return render(request, 'index.html')
