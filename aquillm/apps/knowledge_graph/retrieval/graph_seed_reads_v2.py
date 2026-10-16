"""Explicit bounded source-data reads; transaction/control SQL is outside this meter."""

from math import isfinite

from django.db import connections

from .direct_seed_sql import bounded_seed_read
from .topology.frontier_contracts_v2 import FrontierBudgetExceeded, FrontierBudgetV2


def seed_query_rows(queryset, *, maximum, budget=None):
    if budget is None:
        return tuple(queryset[:maximum])
    return _metered_seed_rows(
        lambda: tuple(queryset[:maximum]), using=queryset.db, maximum=maximum, budget=budget,
    )


def _seed_statement_rows(sql, params, *, using, budget, check):
    """The private native policy statement shares the exact seed data meter."""
    def read():
        check()
        with connections[using].cursor() as cursor:
            check()
            cursor.execute(sql, params)
            return tuple(cursor.fetchall())

    return _metered_seed_rows(read, using=using, maximum=1, budget=budget)


def _metered_seed_rows(
    read, *, using, maximum, budget, deadline=None, clock=None, alias=False,
    after_read=None,
):
    if type(budget) is not FrontierBudgetV2 or type(maximum) is not int or maximum < 1:
        raise TypeError(
            "known seed read requires exact budget and positive cardinality"
        )
    if maximum > budget.max_source_rows:
        raise FrontierBudgetExceeded("seed read cardinality exceeds source allowance")
    if (
        (deadline is not None and (
            type(deadline) not in (int, float) or not isfinite(deadline)
        ))
        or type(alias) is not bool
        or (clock is not None and clock is not budget.clock)
    ):
        raise ValueError("seed read must retain original clock and finite bounds")
    # The supplied QuerySet is the single explicit application-data statement.
    # SET LOCAL, savepoints and restoration remain executable outside the meter.
    with bounded_seed_read(
        using=using,
        deadline=(
            budget.deadline if deadline is None else min(deadline, budget.deadline)
        ),
        clock=budget.clock,
        alias=alias,
    ):
        budget.reserve_read(maximum)
        observed = []

        def capture(execute, sql, params, many, context):
            if observed or many:
                budget.stop_uncertain()
                raise ValueError("seed read dispatched more than one data statement")
            try:
                result = execute(sql, params, many, context)
            except BaseException:
                budget.stop_uncertain()
                raise
            count = context["cursor"].rowcount
            if type(count) is not int or count < 0:
                budget.stop_uncertain()
                raise ValueError("seed driver omitted actual returned-row accounting")
            observed.append(count)
            admitted = budget.charge_read(reserved_rows=maximum, scanned_rows=count)
            if not admitted or count > maximum:
                raise ValueError("seed source exceeded its reserved allowance")
            budget.check_active()
            return result

        try:
            with connections[using].execute_wrapper(capture):
                rows = read()
            if len(observed) != 1 or len(rows) != observed[0]:
                budget.stop_uncertain()
                raise ValueError("seed data read lacks exact physical accounting")
        except BaseException:
            if not observed:
                budget.stop_uncertain()
            raise
        budget.check_active()
        if after_read is not None:
            after_read()
        return rows


_NATIVE_SEED_METER = _metered_seed_rows
_NATIVE_SEED_BOUND = bounded_seed_read
