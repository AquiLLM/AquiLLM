"""Build querysets and CSV rows for feedback / rating export."""
from __future__ import annotations

import csv
import io
from datetime import datetime, time
from datetime import timezone as py_tz
from typing import Any, Iterator

from django.db.models import Count, IntegerField, OuterRef, Q, Subquery
from django.db.models.functions import Coalesce
from django.utils import timezone as dj_tz
from django.utils.dateparse import parse_date, parse_datetime

from apps.chat.models import Message


def parse_query_bounds(
    start_date: str | None, end_date: str | None
) -> tuple[datetime | None, datetime | None]:
    """Parse optional start/end from YYYY-MM-DD or ISO datetimes (aware when possible)."""
    start = end = None
    if start_date:
        if (p := parse_date(start_date)) is not None:
            start = dj_tz.make_aware(datetime.combine(p, time.min))
        elif (pd := parse_datetime(start_date)) is not None:
            start = pd if dj_tz.is_aware(pd) else dj_tz.make_aware(pd)
    if end_date:
        if (p := parse_date(end_date)) is not None:
            end = dj_tz.make_aware(datetime.combine(p, time.max))
        elif (pd := parse_datetime(end_date)) is not None:
            end = pd if dj_tz.is_aware(pd) else dj_tz.make_aware(pd)
    return start, end


def feedback_export_queryset(
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_rating: int | None = None,
    user_number: int | None = None,
):
    user_cnt_subq = (
        Message.objects.filter(
            conversation_id=OuterRef("conversation_id"),
            role="user",
            sequence_number__lte=OuterRef("sequence_number"),
        )
        .values("conversation_id")
        .annotate(c=Count("id"))
        .values("c")
    )

    qs = (
        Message.objects.filter(role="assistant")
        .filter(
            Q(rating__isnull=False)
            | (Q(feedback_text__isnull=False) & ~Q(feedback_text=""))
        )
        .select_related("conversation", "conversation__owner")
        .annotate(
            effective_date=Coalesce("feedback_submitted_at", "created_at"),
            question_number=Subquery(user_cnt_subq, output_field=IntegerField()),
        )
    )

    if start_date is not None:
        qs = qs.filter(effective_date__gte=start_date)
    if end_date is not None:
        qs = qs.filter(effective_date__lte=end_date)
    if min_rating is not None:
        qs = qs.filter(rating__gte=min_rating)
    if user_number is not None:
        qs = qs.filter(conversation__owner_id=user_number)

    return qs.order_by("effective_date", "id")


def _to_iso_utc_z(dt: datetime) -> str:
    if dj_tz.is_naive(dt):
        dt = dj_tz.make_aware(dt, dj_tz.get_current_timezone())
    utc = dt.astimezone(py_tz.utc)
    s = utc.strftime("%Y-%m-%dT%H:%M:%S")
    if utc.microsecond:
        frac = f"{utc.microsecond:06d}".rstrip("0")
        if frac:
            s += f".{frac}"
    return s + "Z"


def iter_feedback_csv_rows(qs) -> Iterator[tuple[str, str, str, str, str]]:
    for m in qs:
        submitted = m.feedback_submitted_at or m.created_at
        date_s = _to_iso_utc_z(submitted)
        user_num = str(m.conversation.owner_id)
        rating_s = "" if m.rating is None else str(m.rating)
        qn = str(m.question_number or 0)
        comments = m.feedback_text or ""
        yield (date_s, user_num, rating_s, qn, comments)


def stream_feedback_csv_lines(
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_rating: int | None = None,
    user_number: int | None = None,
) -> Iterator[str]:
    qs = feedback_export_queryset(
        start_date=start_date,
        end_date=end_date,
        min_rating=min_rating,
        user_number=user_number,
    )
    header_buf = io.StringIO()
    csv.writer(header_buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n").writerow(
        ["date", "user_number", "rating", "question_number", "comments"]
    )
    yield header_buf.getvalue()
    for row in iter_feedback_csv_rows(qs):
        line = io.StringIO()
        csv.writer(line, quoting=csv.QUOTE_MINIMAL, lineterminator="\n").writerow(list(row))
        yield line.getvalue()
