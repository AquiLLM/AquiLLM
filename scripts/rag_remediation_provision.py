"""Provision only CI's projection roles in the disposable PostgreSQL database."""
import os
import psycopg2
from psycopg2 import sql
from scripts.run_rag_remediation_tests import validate_environment


def main():
    validate_environment(os.environ)
    connection = psycopg2.connect(host="db", dbname="rag_test", user="rag_test", password="rag-synthetic-password")
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for role in ("aquillm_projection_source", "aquillm_projection_state"):
                cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
                if cursor.fetchone() is None:
                    cursor.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
                # Existing owned volumes may retain the earlier NOLOGIN roles.
                # Migration-owned table/function grants remain unchanged.
                cursor.execute(sql.SQL(
                    "ALTER ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                    "NOREPLICATION NOBYPASSRLS PASSWORD %s"
                ).format(sql.Identifier(role)), ("rag-synthetic-password",))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
