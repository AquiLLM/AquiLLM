"""
Views for Zotero OAuth and sync functionality
"""
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ObjectDoesNotExist

from .models import ZoteroConnection
from .zotero_oauth import ZoteroOAuthClient
from .zotero_tasks import sync_zotero_library

import logging

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET"])
def zotero_settings(request: HttpRequest) -> HttpResponse:
    """
    Display Zotero connection settings page.

    Shows connection status and provides options to connect, disconnect, or sync.
    """
    try:
        connection = ZoteroConnection.objects.get(user=request.user)
        connected = True
    except ZoteroConnection.DoesNotExist:
        connection = None
        connected = False

    context = {
        'connected': connected,
        'connection': connection,
    }

    return render(request, 'zotero/settings.html', context)


@login_required
@require_http_methods(["GET"])
def zotero_connect(request: HttpRequest) -> HttpResponse:
    """
    Initiate OAuth flow to connect Zotero account (OOB flow).

    Step 1: Get authorization URL and redirect user to Zotero.
    User will receive a verification code to copy/paste back.
    """
    try:
        # Create OAuth client
        oauth_client = ZoteroOAuthClient()

        # Define permissions to request
        permissions = {
            'name': 'AquiLLM',
            'library_access': '1',  # Read access to library
            'notes_access': '1',    # Read access to notes
            'write_access': '0',    # No write access needed
            'all_groups': 'read'    # Read access to groups
        }

        # Get authorization URL with OOB callback
        auth_url, oauth_token, oauth_token_secret = oauth_client.get_authorization_url(
            callback_url='oob',  # Out-of-band for client apps
            permissions=permissions
        )

        # Store token and secret in session for verification step
        request.session['zotero_oauth_token'] = oauth_token
        request.session['zotero_oauth_token_secret'] = oauth_token_secret

        # Store the auth URL to display to user
        request.session['zotero_auth_url'] = auth_url

        # Redirect to verification page instead of directly to Zotero
        return redirect('zotero_verify')

    except Exception as e:
        logger.error(f"Error initiating Zotero OAuth: {str(e)}")
        messages.error(request, f"Failed to connect to Zotero: {str(e)}")
        return redirect('zotero_settings')


@login_required
@require_http_methods(["GET", "POST"])
def zotero_verify(request: HttpRequest) -> HttpResponse:
    """
    Handle OOB verification code entry.

    GET: Display form with Zotero authorization link and verification code input
    POST: Process verification code and complete OAuth flow
    """
    if request.method == "GET":
        # Show verification page with link to Zotero and input for code
        auth_url = request.session.get('zotero_auth_url')
        if not auth_url:
            messages.error(request, "OAuth session expired. Please try connecting again.")
            return redirect('zotero_settings')

        context = {
            'auth_url': auth_url
        }
        return render(request, 'zotero/verify.html', context)

    else:  # POST
        try:
            # Get verification code from form
            oauth_verifier = request.POST.get('verification_code', '').strip()

            if not oauth_verifier:
                messages.error(request, "Please enter the verification code from Zotero")
                return redirect('zotero_verify')

            # Get token and secret from session
            oauth_token = request.session.get('zotero_oauth_token')
            oauth_token_secret = request.session.get('zotero_oauth_token_secret')

            if not oauth_token or not oauth_token_secret:
                raise ValueError("OAuth session expired. Please try connecting again.")

            # Exchange for access token
            oauth_client = ZoteroOAuthClient()
            credentials = oauth_client.get_access_token(
                oauth_token=oauth_token,
                oauth_token_secret=oauth_token_secret,
                oauth_verifier=oauth_verifier
            )

            # Store credentials in database
            connection, created = ZoteroConnection.objects.update_or_create(
                user=request.user,
                defaults={
                    'api_key': credentials['api_key'],
                    'zotero_user_id': credentials['user_id']
                }
            )

            # Clean up session
            request.session.pop('zotero_oauth_token', None)
            request.session.pop('zotero_oauth_token_secret', None)
            request.session.pop('zotero_auth_url', None)

            if created:
                messages.success(request, f"Successfully connected Zotero account: {credentials['username']}")
            else:
                messages.success(request, f"Zotero account reconnected: {credentials['username']}")

            return redirect('zotero_settings')

        except Exception as e:
            logger.error(f"Error completing Zotero OAuth: {str(e)}")
            messages.error(request, f"Failed to complete Zotero connection: {str(e)}")
            return redirect('zotero_settings')


@login_required
@require_http_methods(["POST"])
def zotero_disconnect(request: HttpRequest) -> HttpResponse:
    """
    Disconnect Zotero account by removing stored credentials.
    """
    try:
        connection = ZoteroConnection.objects.get(user=request.user)
        connection.delete()
        messages.success(request, "Zotero account disconnected successfully")
    except ZoteroConnection.DoesNotExist:
        messages.warning(request, "No Zotero account connected")

    return redirect('zotero_settings')


@login_required
@require_http_methods(["POST"])
def zotero_sync(request: HttpRequest) -> HttpResponse:
    """
    Trigger background sync of Zotero library.

    Starts a Celery task to sync collections and items from Zotero.
    """
    try:
        # Check if user has Zotero connection
        connection = ZoteroConnection.objects.get(user=request.user)

        # Trigger background sync task
        task = sync_zotero_library.delay(user_id=request.user.id)

        messages.info(request, "Zotero sync started. This may take a few minutes depending on your library size.")
        logger.info(f"Started Zotero sync for user {request.user.id} (task: {task.id})")

        return redirect('zotero_settings')

    except ZoteroConnection.DoesNotExist:
        messages.error(request, "Please connect your Zotero account first")
        return redirect('zotero_settings')
    except Exception as e:
        logger.error(f"Error starting Zotero sync: {str(e)}")
        messages.error(request, f"Failed to start sync: {str(e)}")
        return redirect('zotero_settings')


@login_required
@require_http_methods(["GET"])
def zotero_sync_status(request: HttpRequest) -> JsonResponse:
    """
    Get current sync status (for AJAX polling).

    Returns JSON with sync information.
    """
    try:
        connection = ZoteroConnection.objects.get(user=request.user)
        return JsonResponse({
            'connected': True,
            'last_synced_at': connection.last_synced_at.isoformat() if connection.last_synced_at else None,
            'last_sync_version': connection.last_sync_version
        })
    except ZoteroConnection.DoesNotExist:
        return JsonResponse({'connected': False})
