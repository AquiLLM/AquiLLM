from django.shortcuts import render


def index(request):
    """Main landing page view"""
    context = {
        'page_title': 'AquiLLM - Home',
        'project_name': 'AquiLLM',
        'tagline': 'A Tool for Preserving Knowledge in Research Groups',
        'description': 'AquiLLM (pronounced ah-quill-em) is a tool that helps researchers manage, search, and interact with their research documents. The goal is to enable teams to access and preserve their collective knowledge, and to enable new group members to get up to speed quickly.',
    }
    return render(request, 'index.html', context)
