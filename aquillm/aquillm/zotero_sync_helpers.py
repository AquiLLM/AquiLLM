"""Zotero library tree flattening and parallel fetch helpers for sync UI."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


def flatten_collections(collections: list[dict], items: list[dict]) -> list[dict]:
    """
    Flatten hierarchical collections into a list with depth information.
    Returns list of dicts with 'key', 'name', 'depth', 'parent', 'item_count'
    """
    collection_dict: dict[str, dict[str, Any]] = {}
    flattened: list[dict[str, Any]] = []

    collection_item_counts: dict[str, int] = {}
    for item in items:
        item_collections = item["data"].get("collections", [])
        for col_key in item_collections:
            collection_item_counts[col_key] = collection_item_counts.get(col_key, 0) + 1

    for col in collections:
        col_key = col["key"]
        col_data = col["data"]
        collection_dict[col_key] = {
            "key": col_key,
            "name": col_data["name"],
            "parent": col_data.get("parentCollection"),
            "item_count": collection_item_counts.get(col_key, 0),
        }

    def add_collections_recursive(parent_key: str | None, depth: int) -> None:
        for col_key, col_info in collection_dict.items():
            if col_info["parent"] == parent_key:
                flattened.append(
                    {
                        "key": col_key,
                        "name": col_info["name"],
                        "depth": depth,
                        "parent": parent_key,
                        "item_count": col_info["item_count"],
                    }
                )
                add_collections_recursive(col_key, depth + 1)

    for col_key, col_info in collection_dict.items():
        if not col_info["parent"] or col_info["parent"] not in collection_dict:
            flattened.append(
                {
                    "key": col_key,
                    "name": col_info["name"],
                    "depth": 0,
                    "parent": None,
                    "item_count": col_info["item_count"],
                }
            )
            add_collections_recursive(col_key, 1)

    return flattened


def fetch_library_data(
    client: Any,
    lib_id: str,
    lib_name: str,
    lib_type: str,
    group_id: str | None,
) -> dict[str, Any]:
    """Fetch collections and items for a library (runs Zotero HTTP in a thread pool)."""
    with ThreadPoolExecutor(max_workers=2) as inner_executor:
        collections_future = inner_executor.submit(client.get_collections, group_id=group_id)
        items_future = inner_executor.submit(client.get_top_level_items, group_id=group_id)
        collections = collections_future.result()
        items = items_future.result()
    return {
        "id": lib_id,
        "name": lib_name,
        "type": lib_type,
        "collections": flatten_collections(collections, items),
    }


__all__ = ["fetch_library_data", "flatten_collections"]
