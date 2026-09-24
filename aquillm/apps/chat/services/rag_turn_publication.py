"""Final conversation writes stay cancellable even after retrieval is sealed."""

from contextlib import contextmanager

from django.db import connections, transaction

from lib.llm.turn_context import check_turn_active, current_turn


@contextmanager
def conversation_publication():
    state = current_turn()
    if state is None:
        yield
        return
    check_turn_active()
    connection = connections["default"]
    commit_locked = False
    try:
        with transaction.atomic():
            if connection.vendor == "postgresql":
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '3000ms'")
            yield
            # Keep cancellation live during the write, then serialize the final
            # check AND transaction exit with close to prevent a check/commit race.
            state.budget._lock.acquire()
            commit_locked = True
            check_turn_active()
    finally:
        if commit_locked:
            state.budget._lock.release()
