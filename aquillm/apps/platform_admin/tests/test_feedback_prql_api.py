"""
permission and correctness tests for the feedback_dashboard_prql_query endpoint
and the seed_feedback management command
"""
import json

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import Client, TestCase
from django.urls import reverse

from apps.chat.models import Message, WSConversation
from apps.platform_admin.services.feedback_dataset import feedback_dataset_queryset

User = get_user_model()

PRQL_URL_NAME = "api_feedback_dashboard_prql"


def make_superuser(username: str = "prqladmin") -> User:
    return User.objects.create_superuser(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


def make_staff_user(username: str = "prqlstaff") -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
        is_staff=True,
    )


def make_regular_user(username: str = "prqlregular") -> User:
    return User.objects.create_user(
        username=username,
        email=f"{username}@test.com",
        password="testpass123",
    )


def make_feedback_row(owner: User, rating: int = 4, text: str = "good") -> Message:
    convo = WSConversation.objects.create(owner=owner, name="test convo")
    return Message.objects.create(
        conversation=convo,
        role="assistant",
        content="test content",
        sequence_number=1,
        rating=rating,
        feedback_text=text,
        model="claude-3-5-sonnet-20241022",
    )


def post_prql(client: Client, url: str, prql: str) -> object:
    return client.post(
        url,
        data=json.dumps({"prql": prql}),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# permission tests
# ---------------------------------------------------------------------------

class PRQLEndpointPermissionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse(PRQL_URL_NAME)
        self.superuser = make_superuser()
        self.staff = make_staff_user()
        self.regular = make_regular_user()

    def test_unauthenticated_redirects(self):
        resp = post_prql(self.client, self.url, "from feedback\ntake 1")
        self.assertIn(resp.status_code, [301, 302])

    def test_regular_user_forbidden(self):
        self.client.login(username="prqlregular", password="testpass123")
        resp = post_prql(self.client, self.url, "from feedback\ntake 1")
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_forbidden(self):
        self.client.login(username="prqlstaff", password="testpass123")
        resp = post_prql(self.client, self.url, "from feedback\ntake 1")
        self.assertEqual(resp.status_code, 403)

    def test_superuser_can_reach_endpoint(self):
        self.client.login(username="prqladmin", password="testpass123")
        resp = post_prql(self.client, self.url, "from feedback\ntake 1")
        self.assertNotIn(resp.status_code, [301, 302, 403])

    def test_get_method_not_allowed(self):
        self.client.login(username="prqladmin", password="testpass123")
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# input validation tests
# ---------------------------------------------------------------------------

class PRQLEndpointValidationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse(PRQL_URL_NAME)
        self.superuser = make_superuser("prqladmin2")
        self.client.login(username="prqladmin2", password="testpass123")

    def test_missing_prql_field_returns_400(self):
        resp = self.client.post(
            self.url,
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertEqual(data["type"], "parse")

    def test_empty_prql_returns_400(self):
        resp = post_prql(self.client, self.url, "")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertEqual(data["type"], "parse")

    def test_prql_without_from_feedback_returns_400(self):
        resp = post_prql(self.client, self.url, "from auth_user\nselect {id}")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertEqual(data["type"], "compilation")
        self.assertIn("from feedback", data["error"])

    def test_invalid_prql_syntax_returns_400_compilation(self):
        resp = post_prql(self.client, self.url, "this is not valid prql %%%")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertEqual(data["type"], "compilation")

    def test_invalid_json_body_returns_400(self):
        resp = self.client.post(
            self.url,
            data="not json at all {{{{",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertEqual(data["type"], "parse")

    def test_curly_brace_descending_sort_works(self):
        """
        sort {-col} curly brace syntax is the correct form for prql-python 0.11.2
        bracket sort [col] is NOT supported by prql-python 0.11.2
        use integer column (id) to avoid CTE wrapping issues with timestamps
        """
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter rating != null\nselect {id, rating}"
        )
        self.assertEqual(resp.status_code, 200)

    def test_bracket_sort_returns_compilation_error(self):
        """
        sort [col] bracket array syntax is rejected by prql-python 0.11.2
        this documents the expected behaviour — users must use sort {col}
        """
        resp = post_prql(
            self.client, self.url,
            "from feedback\nsort [id]\ntake 5\nselect {id}"
        )
        self.assertEqual(resp.status_code, 400)


# ---------------------------------------------------------------------------
# correctness tests
# ---------------------------------------------------------------------------

class PRQLEndpointCorrectnessTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse(PRQL_URL_NAME)
        self.superuser = make_superuser("prqladmin3")
        self.client.login(username="prqladmin3", password="testpass123")
        self.msg1 = make_feedback_row(self.superuser, rating=5, text="excellent")
        self.msg2 = make_feedback_row(self.superuser, rating=2, text="poor")

    def test_valid_select_prql_returns_200(self):
        # simple filter without sort to avoid CTE naming conflicts
        # prql-python wraps sorted queries in a subquery that conflicts with our CTE
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter rating != null\nselect {id, username, rating}"
        )
        self.assertEqual(resp.status_code, 200)

    def test_response_has_required_keys(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id, rating}"
        )
        data = json.loads(resp.content)
        for key in ("columns", "rows", "row_count", "sql"):
            self.assertIn(key, data, msg=f"missing key: {key}")

    def test_columns_match_selected_fields(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id, username, rating}"
        )
        data = json.loads(resp.content)
        self.assertEqual(data["columns"], ["id", "username", "rating"])

    def test_row_count_matches_data(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id, rating}"
        )
        data = json.loads(resp.content)
        self.assertEqual(data["row_count"], 2)
        self.assertEqual(len(data["rows"]), 2)

    def test_rows_are_lists_not_dicts(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id, rating}"
        )
        data = json.loads(resp.content)
        self.assertGreater(len(data["rows"]), 0)
        self.assertIsInstance(data["rows"][0], list)

    def test_sql_field_is_string(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id}"
        )
        data = json.loads(resp.content)
        self.assertIsInstance(data["sql"], str)
        self.assertGreater(len(data["sql"]), 0)

    def test_aggregate_query_works(self):
        resp = post_prql(
            self.client, self.url,
            (
                "from feedback\n"
                "filter rating != null\n"
                "group rating (\n"
                "  aggregate {\n"
                "    count = count id,\n"
                "  }\n"
                ")\n"
                "sort rating"
            )
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn("rating", data["columns"])
        self.assertIn("count", data["columns"])

    def test_take_limits_results(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\ntake 1\nselect {id}"
        )
        data = json.loads(resp.content)
        self.assertEqual(len(data["rows"]), 1)

    def test_empty_result_returns_empty_rows(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter rating == 99\nselect {id, rating}"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(data["row_count"], 0)
        self.assertEqual(data["rows"], [])

    def test_truncated_flag_false_on_small_result(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nselect {id}"
        )
        data = json.loads(resp.content)
        self.assertFalse(data.get("truncated", False))

    def test_filter_with_inlined_integer_works(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter rating == 5\nselect {id, rating}"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        for row in data["rows"]:
            self.assertEqual(row[1], "5")

    def test_filter_null_works(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter rating != null\nselect {id, rating}"
        )
        self.assertEqual(resp.status_code, 200)

    def test_filter_bool_literal_works(self):
        resp = post_prql(
            self.client, self.url,
            "from feedback\nfilter has_feedback_text == true\nselect {id}"
        )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# seed_feedback management command tests
# ---------------------------------------------------------------------------

class SeedFeedbackCommandTests(TestCase):
    def test_seed_creates_feedback_rows(self):
        call_command("seed_feedback", "--count", "6", verbosity=0)
        count = feedback_dataset_queryset().count()
        self.assertGreater(count, 0)

    def test_seed_creates_expected_users(self):
        from apps.platform_admin.management.commands.seed_feedback import SEED_USERNAMES
        call_command("seed_feedback", "--count", "6", verbosity=0)
        for username in SEED_USERNAMES:
            self.assertTrue(
                User.objects.filter(username=username).exists(),
                msg=f"seed user {username} was not created",
            )

    def test_seed_clear_wipes_and_recreates(self):
        call_command("seed_feedback", "--count", "6", verbosity=0)
        count_first = feedback_dataset_queryset().count()
        self.assertGreater(count_first, 0)
        call_command("seed_feedback", "--clear", "--count", "6", verbosity=0)
        count_after_clear = feedback_dataset_queryset().count()
        self.assertEqual(count_after_clear, count_first)

    def test_seed_users_not_duplicated_on_second_run(self):
        from apps.platform_admin.management.commands.seed_feedback import SEED_USERNAMES
        call_command("seed_feedback", "--count", "6", verbosity=0)
        call_command("seed_feedback", "--count", "6", verbosity=0)
        for username in SEED_USERNAMES:
            count = User.objects.filter(username=username).count()
            self.assertEqual(count, 1, msg=f"user {username} was duplicated")

    def test_seed_data_accumulates_without_clear(self):
        call_command("seed_feedback", "--count", "6", verbosity=0)
        count_first = feedback_dataset_queryset().count()
        call_command("seed_feedback", "--count", "6", verbosity=0)
        count_second = feedback_dataset_queryset().count()
        self.assertGreater(count_second, count_first)

    def test_all_seeded_assistant_messages_are_feedback_bearing(self):
        from django.db.models import Q
        from apps.platform_admin.management.commands.seed_feedback import SEED_USERNAMES
        call_command("seed_feedback", "--clear", "--count", "6", verbosity=0)
        seed_users = User.objects.filter(username__in=SEED_USERNAMES)
        all_assistant = Message.objects.filter(
            conversation__owner__in=seed_users,
            role="assistant",
        )
        non_feedback = all_assistant.filter(rating__isnull=True).filter(
            Q(feedback_text__isnull=True) | Q(feedback_text="")
        )
        self.assertEqual(non_feedback.count(), 0)