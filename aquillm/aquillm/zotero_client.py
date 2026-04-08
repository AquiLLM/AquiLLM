"""
Zotero API client for syncing library data
"""
import requests
from requests import Response
from typing import Tuple, List, Dict, Optional, BinaryIO
import structlog

logger = structlog.stdlib.get_logger(__name__)


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

    def _get(self, endpoint: str, full_path: bool = False, params: Dict | None = None) -> requests.Response:
        """
        Make a GET request to the Zotero API.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Optional query parameters

        Returns:
            Response object
        """
        url = endpoint if full_path else f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error("obs.zotero.api_error", url=url, error=str(e))
            raise

    def _next_page(self, res: Response, params: Dict | None = None) -> Response | None:
        if next := res.links.get('next'):
            return self._get(next['url'], full_path=True, params=params)
        return None

    def _get_paginated(self, endpoint: str, params: Dict | None = None) -> Tuple[List, Response]:
        res = self._get(endpoint, params=params)
        ret = res.json()
        while next_res := self._next_page(res, params):
            ret += next_res.json()
            res = next_res
        return ret, res
    

    def get_user_groups(self) -> List[Dict]:
        """
        Fetch all groups the user has access to.

        Returns:
            List of group objects with id, name, etc.
        """
        endpoint = f"/users/{self.user_id}/groups"

        try:
            groups, _ =  self._get_paginated(endpoint)
            logger.info("obs.zotero.fetch_groups", count=len(groups))
            return groups
        except Exception as e:
            logger.error("obs.zotero.api_error", operation="fetch_groups", error=str(e))
            raise

    def get_collections(self, group_id: str | None = None) -> List[Dict]:
        """
        Fetch all collections from the user's library or a group library.

        Args:
            group_id: Optional group ID to fetch group collections instead of user collections

        Returns:
            List of collection objects
        """
        if group_id:
            endpoint = f"/groups/{group_id}/collections"
        else:
            endpoint = f"/users/{self.user_id}/collections"

        try:
            collections, _ = self._get_paginated(endpoint)
            library_type = "group" if group_id else "user"
            logger.info("obs.zotero.fetch_collections", count=len(collections), library_type=library_type)
            return collections
        except Exception as e:
            logger.error("obs.zotero.api_error", operation="fetch_collections", error=str(e))
            raise

    def get_items(self, collection_key: str | None = None, group_id: str | None = None) -> List[Dict]:
        """
        Fetch items from the user's library or a group library.

        Args:
            collection_key: Optional collection key to filter items
            group_id: Optional group ID to fetch group items instead of user items

        Returns:
            List of items
        """
        if group_id:
            if collection_key:
                endpoint = f"/groups/{group_id}/collections/{collection_key}/items"
            else:
                endpoint = f"/groups/{group_id}/items"
        else:
            if collection_key:
                endpoint = f"/users/{self.user_id}/collections/{collection_key}/items"
            else:
                endpoint = f"/users/{self.user_id}/items"

        try:
            items, _ = self._get_paginated(endpoint)

            library_type = "group" if group_id else "user"
            logger.info("obs.zotero.fetch_items", count=len(items), library_type=library_type)
            return items
        except Exception as e:
            logger.error("obs.zotero.api_error", operation="fetch_items", error=str(e))
            raise

    def get_top_level_items(self, group_id: str | None = None) -> List[Dict]:
        """
        Fetch only top-level items (excludes notes, attachments, etc.).

        Args:
            group_id: Optional group ID to fetch group items instead of user items

        Returns:
            List of top-level items
        """
        if group_id:
            endpoint = f"/groups/{group_id}/items/top"
        else:
            endpoint = f"/users/{self.user_id}/items/top"

        try:
            items, _ = self._get_paginated(endpoint)

            library_type = "group" if group_id else "user"
            logger.info("obs.zotero.fetch_top_items", count=len(items), library_type=library_type)
            return items
        except Exception as e:
            logger.error("obs.zotero.api_error", operation="fetch_top_items", error=str(e))
            raise

    def get_item_children(self, item_key: str, group_id: str | None = None) -> List[Dict]:
        """
        Get child items (notes, attachments) for a specific item.

        Args:
            item_key: The key of the parent item
            group_id: Optional group ID if item is in a group library

        Returns:
            List of child items
        """
        if group_id:
            endpoint = f"/groups/{group_id}/items/{item_key}/children"
        else:
            endpoint = f"/users/{self.user_id}/items/{item_key}/children"

        try:
            children, response = self._get_paginated(endpoint)
            logger.info("obs.zotero.fetch_children", count=len(children), item_key=item_key)
            return children
        except Exception as e:
            logger.error("obs.zotero.api_error", operation="fetch_children", item_key=item_key, error=str(e))
            raise

    def download_file(self, item_key: str, group_id: str | None = None) -> Optional[bytes]:
        """
        Download an attached file (PDF, etc.) from Zotero.

        Args:
            item_key: The key of the attachment item
            group_id: Optional group ID if item is in a group library

        Returns:
            File content as bytes, or None if file not available
        """
        if group_id:
            endpoint = f"/groups/{group_id}/items/{item_key}/file"
        else:
            endpoint = f"/users/{self.user_id}/items/{item_key}/file"

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url)
            if response.status_code == 200:
                logger.info("obs.zotero.download_file", item_key=item_key)
                return response.content
            elif response.status_code == 404:
                logger.warning("obs.zotero.download_not_found", item_key=item_key)
                return None
            else:
                response.raise_for_status()
                return None
        except requests.exceptions.RequestException as e:
            logger.error("obs.zotero.download_error", item_key=item_key, error=str(e))
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
                logger.info("obs.zotero.fulltext_empty", item_key=item_key)
                return None
            raise
        except Exception as e:
            logger.error("obs.zotero.fulltext_error", item_key=item_key, error=str(e))
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
                logger.warning("obs.zotero.collection_not_found", collection_key=collection_key)
                return None
            raise
        except Exception as e:
            logger.error("obs.zotero.collection_error", collection_key=collection_key, error=str(e))
            return None
