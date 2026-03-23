"""Compatibility re-exports; prefer ``apps.ingestion.consumers``."""
from apps.ingestion.consumers import IngestMonitorConsumer, IngestionDashboardConsumer

__all__ = ["IngestMonitorConsumer", "IngestionDashboardConsumer"]
