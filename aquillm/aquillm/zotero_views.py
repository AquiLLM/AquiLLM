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
from .zotero_client import ZoteroAPIClient
from .zotero_sync_helpers import fetch_library_data

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
    Initiate OAuth flow to connect Zotero account.

    Step 1: Get authorization URL and redirect user to Zotero.
    """
    try:
        # Create OAuth client
        oauth_client = ZoteroOAuthClient()

        # Build callback URL
        callback_url = request.build_absolute_uri(reverse('zotero_callback'))

        # Define permissions to request
        permissions = {
            'name': 'AquiLLM',
            'library_access': '1',  # Read access to library
            'notes_access': '1',    # Read access to notes
            'write_access': '0',    # No write access needed
            'all_groups': 'read'    # Read access to groups
        }

        # Get authorization URL
        auth_url, oauth_token, oauth_token_secret = oauth_client.get_authorization_url(
            callback_url=callback_url,
            permissions=permissions
        )

        # Store token secret in session for callback
        request.session['zotero_oauth_token'] = oauth_token
        request.session['zotero_oauth_token_secret'] = oauth_token_secret

        # Redirect user to Zotero authorization page
        return redirect(auth_url)

    except Exception as e:
        logger.error(f"Error initiating Zotero OAuth: {str(e)}")
        messages.error(request, f"Failed to connect to Zotero: {str(e)}")
        return redirect('zotero_settings')


@login_required
@require_http_methods(["GET"])
def zotero_callback(request: HttpRequest) -> HttpResponse:
    """
    Handle OAuth callback from Zotero.

    Step 2: Exchange authorization code for access token and store credentials.
    """
    try:
        # Get OAuth verifier from query parameters
        oauth_verifier = request.GET.get('oauth_verifier')
        oauth_token = request.GET.get('oauth_token')

        if not oauth_verifier or not oauth_token:
            raise ValueError("Missing OAuth verifier or token")

        # Get token secret from session
        oauth_token_secret = request.session.get('zotero_oauth_token_secret')
        if not oauth_token_secret:
            raise ValueError("OAuth token secret not found in session")

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
@require_http_methods(["GET", "POST"])
def zotero_sync(request: HttpRequest) -> HttpResponse:
    """
    Display library selection page (GET) or trigger background sync (POST).

    GET: Show available libraries for user to select
    POST: Start sync task with selected libraries
    """
    try:
        # Check if user has Zotero connection
        connection = ZoteroConnection.objects.get(user=request.user)

        if request.method == "GET":
            # Fetch available libraries and their collections
            client = ZoteroAPIClient(
                api_key=connection.api_key,
                user_id=connection.zotero_user_id
            )

            try:
                groups = client.get_user_groups()

                # Sequential library fetches (Zotero API only in threads inside fetch_library_data;
                # no Django ORM on worker threads).
                libraries = []
                try:
                    libraries.append(
                        fetch_library_data(client, "personal", "Personal Library", "user", None)
                    )
                except Exception as e:
                    logger.error("Error fetching personal library: %s", e)

                for group in groups:
                    group_id = str(group["id"])
                    try:
                        libraries.append(
                            fetch_library_data(
                                client, group_id, group["data"]["name"], "group", group_id
                            )
                        )
                    except Exception as e:
                        logger.error("Error fetching library %s: %s", group_id, e)

                # Sort so personal library appears first
                libraries.sort(key=lambda lib: (0 if lib['id'] == 'personal' else 1, lib['name']))

                context = {
                    'libraries': libraries,
                    'connection': connection
                }
                return render(request, 'zotero/sync.html', context)

            except Exception as e:
                logger.error(f"Error fetching Zotero libraries: {str(e)}")
                messages.error(request, f"Failed to fetch libraries: {str(e)}")
                return redirect('zotero_settings')

        else:  # POST
            # Get selected collection keys from form
            # Format: "library_id:collection_key" or "library_id:ALL" for entire library
            selected_items = request.POST.getlist('collections')

            if not selected_items:
                messages.warning(request, "Please select at least one collection to sync")
                return redirect('zotero_sync')

            # Parse selections into library_config
            # Format: {'personal': ['col1', 'col2'], 'group_123': ['ALL'], ...}
            library_config = {}
            for item in selected_items:
                lib_id, col_key = item.split(':', 1)
                if lib_id not in library_config:
                    library_config[lib_id] = []
                library_config[lib_id].append(col_key)

            # Trigger background sync task with collection selection
            task = sync_zotero_library.delay(
                user_id=request.user.id,
                library_config=library_config
            )

            messages.info(request, "Zotero sync started. This may take a few minutes depending on your library size.")
            logger.info(f"Started Zotero sync for user {request.user.id} with collections: {library_config} (task: {task.id})")

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
        })
    except ZoteroConnection.DoesNotExist:
        return JsonResponse({'connected': False})
