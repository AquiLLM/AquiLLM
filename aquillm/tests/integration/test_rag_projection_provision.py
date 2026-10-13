"""Owned role provisioning: real psycopg SQL composition, no database dispatch."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from psycopg2 import sql
from scripts.run_rag_remediation_tests import IsolationError, test_environment as harness_environment

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "rag_projection_provision", ROOT / "scripts/rag_remediation_provision.py"
)
PROVISION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVISION)
ROLES = ("aquillm_projection_source", "aquillm_projection_state")


def composed_parts(statement):
    """Inspect real Composed objects without a live connection or SQL rendering."""
    assert isinstance(statement, sql.Composed)
    return tuple(statement.seq)


class ProjectionProvisionTests(unittest.TestCase):
    def test_new_existing_and_mixed_roles_receive_only_exact_restricted_logins(self):
        for existing in ((False, False), (True, True), (False, True)):
            with self.subTest(existing=existing):
                connection, cursor = MagicMock(), MagicMock()
                connection.cursor.return_value.__enter__.return_value = cursor
                cursor.fetchone.side_effect = [(1,) if found else None for found in existing]
                with patch.dict(os.environ, harness_environment(), clear=True):
                    with patch.object(PROVISION.psycopg2, "connect", return_value=connection) as connect:
                        PROVISION.main()
                connect.assert_called_once_with(host="db", dbname="rag_test", user="rag_test",
                                                password="rag-synthetic-password")
                self.assertIs(connection.autocommit, True)
                connection.close.assert_called_once_with()
                calls = iter(cursor.execute.call_args_list)
                for role, found in zip(ROLES, existing, strict=True):
                    self.assertEqual(next(calls).args, (
                        "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)))
                    if not found:
                        call = next(calls)
                        self.assertEqual(len(call.args), 1)
                        parts = composed_parts(call.args[0])
                        self.assertEqual(parts[0].string, "CREATE ROLE ")
                        self.assertIsInstance(parts[1], sql.Identifier)
                        self.assertEqual(parts[1].strings, (role,))
                        self.assertEqual(parts[2].string, " NOLOGIN")
                    call = next(calls)
                    self.assertEqual(len(call.args), 2)
                    self.assertEqual(call.args[1], ("rag-synthetic-password",))
                    parts = composed_parts(call.args[0])
                    self.assertEqual(parts[0].string, "ALTER ROLE ")
                    self.assertIsInstance(parts[1], sql.Identifier)
                    self.assertEqual(parts[1].strings, (role,))
                    self.assertEqual(parts[2].string,
                        " WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                        "NOREPLICATION NOBYPASSRLS PASSWORD %s")
                self.assertEqual(list(calls), [])  # No grants, ownership changes or extra roles.

    def test_fixed_environment_refuses_foreign_role_password_and_target_before_connect(self):
        changes = (
            ("POSTGRES_HOST", "foreign"),
            ("POSTGRES_NAME", "other_database"),
            ("POSTGRES_PASSWORD", "other_password"),
            ("KG_PROJECTION_POSTGRES_SOURCE_DSN", "postgresql://other:secret@db:5432/rag_test"),
            ("KG_PROJECTION_POSTGRES_STATE_DSN", "postgresql://aquillm_projection_state:other@db:5432/rag_test"),
        )
        for key, value in changes:
            with self.subTest(key=key), patch.dict(os.environ, {**harness_environment(), key: value}, clear=True):
                with patch.object(PROVISION.psycopg2, "connect") as connect:
                    with self.assertRaises(IsolationError):
                        PROVISION.main()
                    connect.assert_not_called()

    def test_role_update_failure_closes_connection_and_propagates(self):
        connection, cursor = MagicMock(), MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = (1,)
        cursor.execute.side_effect = [None, RuntimeError("synthetic role failure")]
        with patch.dict(os.environ, harness_environment(), clear=True):
            with patch.object(PROVISION.psycopg2, "connect", return_value=connection):
                with self.assertRaisesRegex(RuntimeError, "synthetic role failure"):
                    PROVISION.main()
        self.assertEqual(cursor.execute.call_count, 2)
        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
