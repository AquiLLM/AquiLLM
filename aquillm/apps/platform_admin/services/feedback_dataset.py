"""
feedback analytics dataset layer

this is the single source of truth for all feedback analytics queries

dataset contract
----------------
a message row is feedback-bearing if:
    rating is not null
    or (feedback_text is not null and feedback_text != '')

all roles are included (user, assistant, tool) because any role could
hypothetically/theoretically carry a rating or feedback_text in the data model,
role is exposed as a filterable field so superusers can slice by it

canonical annotated fields added to every queryset row
    effective_date      feedback_submitted_at if set, else created_at
    content_snippet     first 300 chars of content for table display
    has_feedback_text   bool, true when feedback_text is non-empty
    conversation_name   alias for conversation__name
    user_id             alias for conversation__owner_id
    username            alias for conversation__owner__username
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from django.db.models import (
    BooleanField,
    Case,
    F,
    Q,
    QuerySet,
    Value,
    When,
)
from django.db.models.functions import Coalesce, Left

from apps.chat.models import Message


# first 300 characters of content are shown as a snippet for table rows
CONTENT_SNIPPET_LENGTH = 300


def feedback_dataset_queryset() -> QuerySet:
    """
    base queryset for all feedback analytics

    returns annotated Message rows where at least one of:
        rating is not null
        feedback_text is non-empty

    all of the consumers must start from this function, never build
    their own separate queryset against Message for dashboard purposes
    """
    return (
        Message.objects.filter(
            Q(rating__isnull=False)
            | (Q(feedback_text__isnull=False) & ~Q(feedback_text=""))
        )
        .select_related("conversation", "conversation__owner")
        .annotate(
            # effective_date is the primary timestamp used for date filtering
            # and sorting, i prefer feedback_submitted_at over created_at
            effective_date=Coalesce("feedback_submitted_at", "created_at"),

            # content_snippet avoids pulling huge content blobs into memory
            # when the table only needs a little bit
            content_snippet=Left("content", CONTENT_SNIPPET_LENGTH),

            # has_feedback_text is a boolean for easy filtering
            # without repeating the isnull and empty-string logic everywhere
            has_feedback_text=Case(
                When(
                    Q(feedback_text__isnull=False) & ~Q(feedback_text=""),
                    then=Value(True),
                ),
                default=Value(False),
                output_field=BooleanField(),
            ),

            # flat aliases so consumers will never need double underscore traversal
            # after annotating, m.username works instead of m.conversation.owner.username
            conversation_name=F("conversation__name"),
            user_id=F("conversation__owner_id"),
            username=F("conversation__owner__username"),
        )
        .order_by("effective_date", "id")
    )


class FeedbackFilters:
    """
    structured filter specification for the feedback dataset

    all fields are optional, none means no filter applied,
    build one of these from request params and it passes to
    apply_filters() or get_filtered_queryset()

    fields
    ------
    start_date              include rows where effective_date >= start_date
    end_date                include rows where effective_date <= end_date
    user_id                 filter to a specific user by django auth pk
    min_rating              include rows where rating >= min_rating
    max_rating              include rows where rating <= max_rating
    exact_rating            include rows where rating == exact_rating,
                            takes precedence over min_rating and max_rating
    feedback_text_search    case-insensitive substring on feedback_text
    conversation_name_search case-insensitive substring on conversation name
    role                    filter to a specific role string
    model                   filter to a specific model string
    tool_call_name          filter to a specific tool_call_name string
    has_feedback_text       true = only rows with text, false = only without,
                            none = no filter
    """

    def __init__(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
        exact_rating: int | None = None,
        feedback_text_search: str | None = None,
        conversation_name_search: str | None = None,
        role: str | None = None,
        model: str | None = None,
        tool_call_name: str | None = None,
        has_feedback_text: bool | None = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.user_id = user_id
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.exact_rating = exact_rating
        self.feedback_text_search = feedback_text_search
        self.conversation_name_search = conversation_name_search
        self.role = role
        self.model = model
        self.tool_call_name = tool_call_name
        self.has_feedback_text = has_feedback_text

    @classmethod
    def from_request_params(cls, params: dict[str, Any]) -> "FeedbackFilters":
        """
        build a FeedbackFilters from a flat dict of request GET params

        coerces string values to correct types,
        invalid or empty values are treated as none silently,
        call with request.GET.dict() not dict(request.GET) to avoid
        receiving lists instead of strings from the QueryDict
        """
        from apps.platform_admin.services.feedback_export import parse_query_bounds

        start_date, end_date = parse_query_bounds(
            params.get("start_date"),
            params.get("end_date"),
        )

        def _int(key: str) -> int | None:
            v = params.get(key)
            if isinstance(v, list):
                v = v[0] if v else None
            if v in (None, ""):
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        def _str(key: str) -> str | None:
            v = params.get(key)
            if isinstance(v, list):
                v = v[0] if v else None
            if not v or not str(v).strip():
                return None
            return str(v).strip()

        def _bool(key: str) -> bool | None:
            v = params.get(key)
            if isinstance(v, list):
                v = v[0] if v else None
            if v in (None, ""):
                return None
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("true", "1", "yes")

        return cls(
            start_date=start_date,
            end_date=end_date,
            user_id=_int("user_id"),
            min_rating=_int("min_rating"),
            max_rating=_int("max_rating"),
            exact_rating=_int("exact_rating"),
            feedback_text_search=_str("feedback_text_search"),
            conversation_name_search=_str("conversation_name_search"),
            role=_str("role"),
            model=_str("model"),
            tool_call_name=_str("tool_call_name"),
            has_feedback_text=_bool("has_feedback_text"),
        )

    def to_dict(self) -> dict[str, Any]:
        """serialize filters to a plain dict, useful for logging and passing between layers"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "user_id": self.user_id,
            "min_rating": self.min_rating,
            "max_rating": self.max_rating,
            "exact_rating": self.exact_rating,
            "feedback_text_search": self.feedback_text_search,
            "conversation_name_search": self.conversation_name_search,
            "role": self.role,
            "model": self.model,
            "tool_call_name": self.tool_call_name,
            "has_feedback_text": self.has_feedback_text,
        }


def apply_filters(qs: QuerySet, filters: FeedbackFilters) -> QuerySet:
    if filters.start_date is not None:
        qs = qs.filter(effective_date__gte=filters.start_date)

    if filters.end_date is not None:
        qs = qs.filter(effective_date__lte=filters.end_date)

    if filters.user_id is not None:
        qs = qs.filter(conversation__owner_id=filters.user_id)

    # exact_rating takes full precedence, range filters are ignored when it is set
    if filters.exact_rating is not None:
        qs = qs.filter(rating=filters.exact_rating)
    else:
        if filters.min_rating is not None:
            qs = qs.filter(rating__gte=filters.min_rating)
        if filters.max_rating is not None:
            qs = qs.filter(rating__lte=filters.max_rating)

    if filters.feedback_text_search:
        qs = qs.filter(feedback_text__icontains=filters.feedback_text_search)

    if filters.conversation_name_search:
        qs = qs.filter(conversation__name__icontains=filters.conversation_name_search)

    if filters.role:
        qs = qs.filter(role=filters.role)

    if filters.model:
        qs = qs.filter(model=filters.model)

    if filters.tool_call_name:
        qs = qs.filter(tool_call_name=filters.tool_call_name)

    if filters.has_feedback_text is True:
        qs = qs.filter(
            Q(feedback_text__isnull=False) & ~Q(feedback_text="")
        )
    elif filters.has_feedback_text is False:
        qs = qs.filter(
            Q(feedback_text__isnull=True) | Q(feedback_text="")
        )

    return qs


def get_filtered_queryset(filters: FeedbackFilters) -> QuerySet:
    """
    convenience entry point: base dataset with filters applied

    this is what every api view, export, and aggregate function
    should call, not feedback_dataset_queryset() directly
    """
    qs = feedback_dataset_queryset()
    return apply_filters(qs, filters)
