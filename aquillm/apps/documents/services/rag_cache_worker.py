"""At most four isolated cache workers, no queue, finite caller waits."""

from contextvars import Context
from threading import BoundedSemaphore, Event, Thread
from time import monotonic

MAX_WORKERS = 4
_slots = BoundedSemaphore(MAX_WORKERS)


def bounded_cache_job(function, budget, *, seconds=0.4):
    slots = _slots
    if not slots.acquire(blocking=False):
        return False, None
    done, result = Event(), []

    def run():
        try:
            result.append(function())
        except Exception:
            pass
        finally:
            done.set()
            slots.release()

    # No request context or authority is captured by the worker. A stalled DNS
    # resolver/cleanup occupies one of four slots; no backlog can accumulate.
    try:
        Thread(target=Context().run, args=(run,), daemon=True).start()
    except Exception:
        slots.release()
        return False, None
    deadline = monotonic() + seconds
    while True:
        if not budget.can_publish() or monotonic() >= deadline:
            return False, None
        if done.wait(min(0.01, max(0, deadline - monotonic()))):
            if monotonic() >= deadline or not budget.can_publish():
                return False, None
            return (bool(result), result[0] if result else None)
