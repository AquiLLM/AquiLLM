"""Explicit bounded source-data reads; transaction/control SQL is outside this meter."""

from django.db import connections

from .direct_seed_sql import bounded_seed_read
from .topology.frontier_contracts_v2 import FrontierBudgetExceeded, FrontierBudgetV2


def seed_query_rows(queryset, *, maximum, budget=None):
    if budget is None:
        return tuple(queryset[:maximum])
    if type(budget) is not FrontierBudgetV2 or type(maximum) is not int or maximum < 1:
        raise TypeError(
            "known seed read requires exact budget and positive cardinality"
        )
    using = queryset.db
    if maximum > budget.max_source_rows:
        raise FrontierBudgetExceeded("seed read cardinality exceeds source allowance")
    # The supplied QuerySet is the single explicit application-data statement.
    # SET LOCAL, savepoints and restoration remain executable outside the meter.
    with bounded_seed_read(using=using, deadline=budget.deadline, clock=budget.clock):
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
                rows = tuple(queryset[:maximum])
            if len(observed) != 1 or len(rows) != observed[0]:
                budget.stop_uncertain()
                raise ValueError("seed data read lacks exact physical accounting")
        except BaseException:
            if not observed:
                budget.stop_uncertain()
            raise
        budget.check_active()
        return rows
