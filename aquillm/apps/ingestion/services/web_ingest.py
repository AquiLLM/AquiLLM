"""Schedule asynchronous web crawl ingestion."""
from __future__ import annotations

import structlog
from urllib.parse import urlparse

from aquillm.crawler_tasks import crawl_and_ingest_webpage
from aquillm.metrics import ingestion_items

logger = structlog.stdlib.get_logger(__name__)


def schedule_webpage_crawl(url: str, collection_id: int, user_id: int, *, max_depth: int) -> None:
    crawl_and_ingest_webpage.delay(url, collection_id, user_id, max_depth=max_depth)
    ingestion_items.labels(source="web", status="queued").inc()
    parsed = urlparse(url)
    logger.info("obs.ingest.web_crawl_dispatched", url_host=parsed.hostname, url_path=parsed.path, collection_id=collection_id)


__all__ = ["schedule_webpage_crawl"]
