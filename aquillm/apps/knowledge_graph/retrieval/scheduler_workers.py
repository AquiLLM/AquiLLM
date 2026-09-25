"""Process-wide bounded worker capacity for cooperative graph branch calls."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore

_MAX_WORKERS = 4


class BoundedWorkerPool:
    """Bound both live workers and queued calls across retrieval requests."""

    def __init__(self) -> None:
        self._slots = BoundedSemaphore(_MAX_WORKERS)
        self._executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="kg-branch",
        )

    def submit(self, function: Callable, *args) -> Future | None:
        submitted = self.submit_batch(((function, args),))
        return None if submitted is None else submitted[0]

    def submit_batch(
        self, calls: tuple[tuple[Callable, tuple], ...]
    ) -> tuple[Future, ...] | None:
        acquired = 0
        for _ in calls:
            if not self._slots.acquire(blocking=False):
                for _ in range(acquired):
                    self._slots.release()
                return None
            acquired += 1
        futures: list[Future] = []
        try:
            for function, args in calls:
                future = self._executor.submit(function, *args)
                future.add_done_callback(self._release)
                futures.append(future)
        except Exception:
            for _ in range(acquired - len(futures)):
                self._slots.release()
            for future in futures:
                future.cancel()
            raise
        return tuple(futures)

    def _release(self, _future: Future) -> None:
        self._slots.release()


BRANCH_WORKERS = BoundedWorkerPool()

__all__ = ["BRANCH_WORKERS"]
