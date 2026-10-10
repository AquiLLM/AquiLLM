"""Provision only CI's projection roles in the disposable PostgreSQL database."""
import os
import psycopg2
from psycopg2 import sql
from scripts.run_rag_remediation_tests import validate_environment

validate_environment(os.environ)
connection = psycopg2.connect(host="db", dbname="rag_test", user="rag_test", password="rag-synthetic-password")
connection.autocommit = True
try:
    with connection.cursor() as cursor:
        for role in ("aquillm_projection_source", "aquillm_projection_state"):
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
finally:
    connection.close()
