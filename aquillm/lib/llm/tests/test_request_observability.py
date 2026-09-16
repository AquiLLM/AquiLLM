"""Safe provider usage and latency observability contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from lib.llm.providers.request_observability import (
    UsageCounts,
    extract_usage,
    log_request_completed,
)


def test_extract_usage_reports_reasoning_when_provider_supplies_it():
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=40,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=31),
    )

    assert extract_usage(usage) == UsageCounts(120, 40, 31)


def test_extract_usage_marks_reasoning_unavailable_instead_of_inventing_zero():
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40)

    assert extract_usage(usage).reasoning_tokens is None


def test_completed_request_log_contains_only_safe_operational_fields():
    logger = MagicMock()
    log_request_completed(
        logger=logger,
        correlation_id="0f22db7309f04ab0a4676cdb5a76f962",
        stage="direct_synthesis",
        configured_max_tokens=4096,
        effective_max_tokens=4096,
        thinking_requested=True,
        usage=UsageCounts(120, 40, None),
        finish_reason="stop",
        request_started_at=10.0,
        first_signal_at=10.2,
        completed_at=11.0,
        tool_name=None,
    )

    event = logger.info.call_args.args[0]
    fields = logger.info.call_args.kwargs
    serialized = json.dumps(fields, sort_keys=True)
    assert event == "obs.llm.request_completed"
    assert fields["reasoning_tokens"] == "unavailable"
    assert fields["ttft_ms"] == 200.0
    assert fields["duration_ms"] == 1000.0
    assert "attensity" not in serialized
    assert "search_string" not in serialized
    assert "document" not in serialized
