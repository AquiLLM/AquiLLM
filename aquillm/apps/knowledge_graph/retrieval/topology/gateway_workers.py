"""Nonqueued gateway reads; abandoned calls retain slots until workers finish."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from time import monotonic


class GatewayWorkerPool:
    def __init__(self):
        self._slots = BoundedSemaphore(4)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kg-gateway")

    async def run(self, function, *args, expires, clock=monotonic, **kwargs):
        started = asyncio.get_running_loop().time()
        remaining = expires - clock()
        if remaining <= 0:
            raise TimeoutError("gateway deadline expired")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("gateway capacity exhausted")

        def read():
            if expires <= clock():
                raise TimeoutError("gateway deadline expired")
            return function(*args, **kwargs)

        try:
            future = self._executor.submit(read)
        except BaseException:
            self._slots.release()
            raise
        # Release on actual worker completion, never on coroutine cancellation.
        future.add_done_callback(lambda _: self._slots.release())
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(_consume_abandoned_exception)
        wait = max(0.0, remaining - (asyncio.get_running_loop().time() - started))
        done, _ = await asyncio.wait((wrapped,), timeout=wait)
        if not done:
            raise TimeoutError("gateway deadline expired")
        result = wrapped.result()
        if expires <= clock():
            raise TimeoutError("gateway deadline expired")
        return result


def _consume_abandoned_exception(future):
    if not future.cancelled():
        future.exception()


GATEWAY_WORKERS = GatewayWorkerPool()
