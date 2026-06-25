"""WebSocket error payloads for chat consumer."""
from __future__ import annotations

import sys
from json import dumps
from typing import Any

from aquillm.settings import DEBUG


async def send_connect_error(consumer: Any, exc: BaseException) -> None:
    if DEBUG:
        from django.views.debug import ExceptionReporter

        reporter = ExceptionReporter(None, *sys.exc_info())
        debug_html = reporter.get_traceback_html()
        await consumer.send(text_data=dumps({"exception": str(exc), "debug_html": debug_html}))
    else:
        await consumer.send(
            text_data='{"exception": "A server error has occurred. Try reloading the page"}'
        )


async def send_receive_validation_error(consumer: Any, message: str) -> None:
    await consumer.send(text_data=dumps({"exception": message}))


async def send_receive_error(consumer: Any, exc: BaseException) -> None:
    if DEBUG:
        from django.views.debug import ExceptionReporter

        reporter = ExceptionReporter(None, *sys.exc_info())
        debug_html = reporter.get_traceback_html()
        await consumer.send(text_data=dumps({"exception": str(exc), "debug_html": debug_html}))
    else:
        await consumer.send(
            text_data='{"exception": "A server error has occurred. Try reloading the page"}'
        )


__all__ = ["send_connect_error", "send_receive_error", "send_receive_validation_error"]
