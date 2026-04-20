"""Schedule asynchronous web crawl ingestion."""
from __future__ import annotations

from aquillm.crawler_tasks import crawl_and_ingest_webpage


def schedule_webpage_crawl(url: str, collection_id: int, user_id: int, *, max_depth: int) -> None:
    crawl_and_ingest_webpage.delay(url, collection_id, user_id, max_depth=max_depth)


__all__ = ["schedule_webpage_crawl"]
