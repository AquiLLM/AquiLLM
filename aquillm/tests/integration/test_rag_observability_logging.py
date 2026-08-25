"""Production logging coverage for privacy-safe RAG telemetry."""

from aquillm.settings_logging import LOGGING


def test_rag_observability_logger_names_emit_info_events():
    for logger_name in ("apps.chat", "apps.documents", "lib.llm.providers"):
        config = LOGGING["loggers"][logger_name]
        assert config["level"] == "INFO"
        assert config["handlers"] == ["console"]
        assert config["propagate"] is False
