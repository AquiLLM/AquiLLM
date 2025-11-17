"""
Zotero OAuth 1.0a client for authentication flow
"""
import os
from typing import Dict, Tuple
from requests_oauthlib import OAuth1Session
import logging

logger = logging.getLogger(__name__)


class ZoteroOAuthClient:
    """
    Handles OAuth 1.0a authentication flow with Zotero.

    OAuth flow:
    1. Get request token from Zotero
    2. Redirect user to Zotero for authorization
    3. Exchange authorized token for access token (API key)
    """

    REQUEST_TOKEN_URL = "https://www.zotero.org/oauth/request"
    AUTHORIZE_URL = "https://www.zotero.org/oauth/authorize"
    ACCESS_TOKEN_URL = "https://www.zotero.org/oauth/access"

    def __init__(self, client_key: str | None = None, client_secret: str | None = None):
        """
        Initialize OAuth client with credentials from environment or parameters.

        Args:
            client_key: Zotero OAuth client key (optional, defaults to env var)
            client_secret: Zotero OAuth client secret (optional, defaults to env var)
        """
        self.client_key = client_key or os.environ.get('ZOTERO_CLIENT_KEY')
        self.client_secret = client_secret or os.environ.get('ZOTERO_CLIENT_SECRET')

        if not self.client_key or not self.client_secret:
            raise ValueError(
                "ZOTERO_CLIENT_KEY and ZOTERO_CLIENT_SECRET must be set in environment "
                "or passed as parameters. Register your app at https://www.zotero.org/oauth/apps"
            )

    def get_authorization_url(self, callback_url: str, permissions: Dict[str, str] | None = None) -> Tuple[str, str, str]:
        """
        Step 1 & 2: Get request token and build authorization URL.

        Args:
            callback_url: URL where Zotero will redirect after authorization
            permissions: Optional dict of permissions to request:
                - name: Key description
                - library_access: Read access to library (1 or 0)
                - notes_access: Read access to notes (1 or 0)
                - write_access: Write access (1 or 0)
                - all_groups: Group access level (none, read, or write)

        Returns:
            Tuple of (authorization_url, oauth_token, oauth_token_secret)
        """
        # Create OAuth session
        oauth = OAuth1Session(
            self.client_key,
            client_secret=self.client_secret,
            callback_uri=callback_url
        )

        try:
            # Fetch request token
            response = oauth.fetch_request_token(self.REQUEST_TOKEN_URL)
            oauth_token = response.get('oauth_token')
            oauth_token_secret = response.get('oauth_token_secret')

            # Build authorization URL with optional permissions
            auth_url = oauth.authorization_url(self.AUTHORIZE_URL)

            # Add permission parameters if provided
            if permissions:
                permission_params = []
                for key, value in permissions.items():
                    permission_params.append(f"{key}={value}")
                if permission_params:
                    separator = '&' if '?' in auth_url else '?'
                    auth_url = f"{auth_url}{separator}{'&'.join(permission_params)}"

            logger.info(f"Generated Zotero authorization URL for callback: {callback_url}")
            return auth_url, oauth_token, oauth_token_secret

        except Exception as e:
            logger.error(f"Error getting Zotero authorization URL: {str(e)}")
            raise

    def get_access_token(self, oauth_token: str, oauth_token_secret: str, oauth_verifier: str) -> Dict[str, str]:
        """
        Step 3: Exchange authorized request token for access token.

        Args:
            oauth_token: Request token from step 1
            oauth_token_secret: Request token secret from step 1
            oauth_verifier: Verifier code from Zotero callback

        Returns:
            Dict containing:
                - api_key: The API key for making Zotero API requests
                - user_id: Zotero user ID
                - username: Zotero username
        """
        # Create OAuth session with request token
        oauth = OAuth1Session(
            self.client_key,
            client_secret=self.client_secret,
            resource_owner_key=oauth_token,
            resource_owner_secret=oauth_token_secret,
            verifier=oauth_verifier
        )

        try:
            # Exchange for access token
            response = oauth.fetch_access_token(self.ACCESS_TOKEN_URL)

            # Extract credentials
            # The oauth_token_secret is the API key used for future requests
            api_key = response.get('oauth_token_secret')
            user_id = response.get('userID')
            username = response.get('username')

            logger.info(f"Successfully obtained Zotero access token for user: {username} (ID: {user_id})")

            return {
                'api_key': api_key,
                'user_id': user_id,
                'username': username
            }

        except Exception as e:
            logger.error(f"Error exchanging Zotero OAuth token: {str(e)}")
            raise
