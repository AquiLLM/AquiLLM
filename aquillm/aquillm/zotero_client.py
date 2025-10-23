"""
Zotero API client for syncing library data
"""
import requests
from typing import List, Dict, Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)


class ZoteroAPIClient:
    """
    Client for interacting with Zotero Web API v3.

    Handles fetching collections, items, and file attachments from a user's Zotero library.
    """

    BASE_URL = "https://api.zotero.org"
    API_VERSION = "3"

    def __init__(self, api_key: str, user_id: str):
        """
        Initialize API client with user credentials.

        Args:
            api_key: Zotero API key
            user_id: Zotero user ID
        """
        self.api_key = api_key
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            'Zotero-API-Key': self.api_key,
            'Zotero-API-Version': self.API_VERSION
        })

    def _get(self, endpoint: str, params: Dict = None) -> requests.Response:
        """
        Make a GET request to the Zotero API.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Optional query parameters

        Returns:
            Response object
        """
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Zotero API request failed: {url} - {str(e)}")
            raise

    def get_collections(self, since_version: int = 0) -> List[Dict]:
        """
        Fetch all collections from the user's library.

        Args:
            since_version: Only return collections modified since this version

        Returns:
            List of collection objects
        """
        endpoint = f"/users/{self.user_id}/collections"
        params = {}
        if since_version > 0:
            params['since'] = since_version

        try:
            response = self._get(endpoint, params=params)
            collections = response.json()
            logger.info(f"Fetched {len(collections)} collections from Zotero")
            return collections
        except Exception as e:
            logger.error(f"Error fetching Zotero collections: {str(e)}")
            raise

    def get_items(self, since_version: int = 0, collection_key: str = None) -> tuple[List[Dict], int]:
        """
        Fetch items from the user's library.

        Args:
            since_version: Only return items modified since this version
            collection_key: Optional collection key to filter items

        Returns:
            Tuple of (items list, latest version number)
        """
        if collection_key:
            endpoint = f"/users/{self.user_id}/collections/{collection_key}/items"
        else:
            endpoint = f"/users/{self.user_id}/items"

        params = {}
        if since_version > 0:
            params['since'] = since_version

        try:
            response = self._get(endpoint, params=params)
            items = response.json()

            # Get the latest library version from response headers
            latest_version = int(response.headers.get('Last-Modified-Version', 0))

            logger.info(f"Fetched {len(items)} items from Zotero (version: {latest_version})")
            return items, latest_version
        except Exception as e:
            logger.error(f"Error fetching Zotero items: {str(e)}")
            raise

    def get_top_level_items(self, since_version: int = 0) -> tuple[List[Dict], int]:
        """
        Fetch only top-level items (excludes notes, attachments, etc.).

        Args:
            since_version: Only return items modified since this version

        Returns:
            Tuple of (items list, latest version number)
        """
        endpoint = f"/users/{self.user_id}/items/top"
        params = {}
        if since_version > 0:
            params['since'] = since_version

        try:
            response = self._get(endpoint, params=params)
            items = response.json()
            latest_version = int(response.headers.get('Last-Modified-Version', 0))

            logger.info(f"Fetched {len(items)} top-level items from Zotero (version: {latest_version})")
            return items, latest_version
        except Exception as e:
            logger.error(f"Error fetching top-level items: {str(e)}")
            raise

    def get_item_children(self, item_key: str) -> List[Dict]:
        """
        Get child items (notes, attachments) for a specific item.

        Args:
            item_key: The key of the parent item

        Returns:
            List of child items
        """
        endpoint = f"/users/{self.user_id}/items/{item_key}/children"

        try:
            response = self._get(endpoint)
            children = response.json()
            logger.info(f"Fetched {len(children)} children for item {item_key}")
            return children
        except Exception as e:
            logger.error(f"Error fetching children for item {item_key}: {str(e)}")
            raise

    def download_file(self, item_key: str) -> Optional[bytes]:
        """
        Download an attached file (PDF, etc.) from Zotero.

        Args:
            item_key: The key of the attachment item

        Returns:
            File content as bytes, or None if file not available
        """
        endpoint = f"/users/{self.user_id}/items/{item_key}/file"
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url)
            if response.status_code == 200:
                logger.info(f"Downloaded file for item {item_key}")
                return response.content
            elif response.status_code == 404:
                logger.warning(f"File not found for item {item_key}")
                return None
            else:
                response.raise_for_status()
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file for item {item_key}: {str(e)}")
            return None

    def get_fulltext(self, item_key: str) -> Optional[str]:
        """
        Get indexed full text content for an item.

        Args:
            item_key: The key of the item

        Returns:
            Full text content or None if not available
        """
        endpoint = f"/users/{self.user_id}/items/{item_key}/fulltext"

        try:
            response = self._get(endpoint)
            fulltext_data = response.json()
            return fulltext_data.get('content', '')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.info(f"No fulltext available for item {item_key}")
                return None
            raise
        except Exception as e:
            logger.error(f"Error fetching fulltext for item {item_key}: {str(e)}")
            return None

    def get_collection_by_key(self, collection_key: str) -> Optional[Dict]:
        """
        Get a specific collection by its key.

        Args:
            collection_key: The collection key

        Returns:
            Collection object or None if not found
        """
        endpoint = f"/users/{self.user_id}/collections/{collection_key}"

        try:
            response = self._get(endpoint)
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Collection {collection_key} not found")
                return None
            raise
        except Exception as e:
            logger.error(f"Error fetching collection {collection_key}: {str(e)}")
            return None
