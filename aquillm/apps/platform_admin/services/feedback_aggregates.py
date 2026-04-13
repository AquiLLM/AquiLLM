"""
aggregate metrics for the feedback dashboard

all functions operate on the canonical dataset from feedback_dataset.py,
never query Message directly from here
"""
from __future__ import annotations

from datetime import datetime, timezone as py_tz
from typing import Any

from django.db.models import Avg, Count, Max, Min, Q

from apps.platform_admin.services.feedback_dataset import (
    FeedbackFilters,
    get_filtered_queryset,
)


def get_summary_metrics(filters: FeedbackFilters) -> dict[str, Any]:
    """
    return the aggregate summary metrics for the current filter state

    runs two queries:
        one for counts, avg, and date range
        one for per-rating counts (the distribution)

    returns a dict with keys:
        total_count         total feedback-bearing rows matching filters
        rated_count         rows that have a non-null rating
        avg_rating          float rounded to 2 places, or none if no rated rows
        rating_distribution dict mapping int rating 1-5 to count
        has_text_count      rows with non-empty feedback_text
        date_min            iso utc string of earliest effective_date, or none
        date_max            iso utc string of latest effective_date, or none
    """
    qs = get_filtered_queryset(filters)

    # single query for scalar aggregates
    agg = qs.aggregate(
        total_count=Count("id"),
        rated_count=Count("id", filter=Q(rating__isnull=False)),
        avg_rating=Avg("rating"),
        has_text_count=Count(
            "id",
            filter=Q(feedback_text__isnull=False) & ~Q(feedback_text=""),
        ),
        date_min=Min("effective_date"),
        date_max=Max("effective_date"),
        # per-rating counts in the same query to avoid a second round trip
        r1=Count("id", filter=Q(rating=1)),
        r2=Count("id", filter=Q(rating=2)),
        r3=Count("id", filter=Q(rating=3)),
        r4=Count("id", filter=Q(rating=4)),
        r5=Count("id", filter=Q(rating=5)),
    )

    avg = agg["avg_rating"]

    return {
        "total_count": agg["total_count"],
        "rated_count": agg["rated_count"],
        "avg_rating": round(float(avg), 2) if avg is not None else None,
        "rating_distribution": {
            1: agg["r1"],
            2: agg["r2"],
            3: agg["r3"],
            4: agg["r4"],
            5: agg["r5"],
        },
        "has_text_count": agg["has_text_count"],
        "date_min": _to_iso(agg["date_min"]),
        "date_max": _to_iso(agg["date_max"]),
    }


def get_filter_options() -> dict[str, Any]:
    """
    return available filter option values for populating ui dropdowns

    always operates on the full unfiltered dataset so dropdowns always
    show the complete universe of available values

    returns a dict with keys:
        users       list of dicts with id and username
        roles       sorted list of role strings
        models      sorted list of non-null model strings
        tool_names  sorted list of non-null tool_call_name strings
        ratings     sorted list of int rating values present in data
    """
    from apps.platform_admin.services.feedback_dataset import feedback_dataset_queryset

    # use the base queryset with no filters so all options are visible
    qs = feedback_dataset_queryset()

    # users: distinct combinations of user_id and username
    # these annotations already exist on the queryset from the base function
    users_qs = (
        qs.values("user_id", "username")
        .distinct()
        .order_by("username")
    )
    user_list = [
        {"id": row["user_id"], "username": row["username"]}
        for row in users_qs
    ]

    roles = sorted(
        set(qs.values_list("role", flat=True).distinct())
    )

    # filter out null and empty strings for models and tool names
    models = sorted(
        v for v in qs.values_list("model", flat=True).distinct() if v
    )

    tool_names = sorted(
        v for v in qs.values_list("tool_call_name", flat=True).distinct() if v
    )

    ratings = sorted(
        v for v in qs.values_list("rating", flat=True).distinct()
        if v is not None
    )

    return {
        "users": user_list,
        "roles": roles,
        "models": models,
        "tool_names": tool_names,
        "ratings": ratings,
    }


def _to_iso(dt: datetime | None) -> str | None:
    """convert a datetime to iso utc string, returns none if input is none"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=py_tz.utc)
    return dt.astimezone(py_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")