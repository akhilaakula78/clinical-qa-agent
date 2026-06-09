"""Guardrail tests — written first (TDD).

Guardrails are the JD's 'kill-switch and policy enforcement'. They must:
  1. Reject any non-SELECT statement.
  2. Reject multiple statements.
  3. Reject PHI columns in SELECT projection.
  4. Inject row limit when missing.
  5. Reject unknown tables / suspicious functions.
"""
from __future__ import annotations

import pytest

from clinical_qa.guardrails.sql_safety import (
    GuardrailViolation,
    enforce_row_limit,
    validate_sql,
)


# ---------- 1. Statement type ----------

def test_select_allowed():
    validate_sql("SELECT COUNT(*) FROM encounters")


@pytest.mark.parametrize("sql", [
    "DROP TABLE encounters",
    "DELETE FROM patients WHERE 1=1",
    "UPDATE patients SET first_name='X'",
    "INSERT INTO patients VALUES (1)",
    "TRUNCATE encounters",
    "ALTER TABLE patients ADD COLUMN x INT",
    "CREATE TABLE x (a INT)",
    "GRANT SELECT ON patients TO bob",
])
def test_destructive_rejected(sql):
    with pytest.raises(GuardrailViolation, match="non-SELECT|destructive"):
        validate_sql(sql)


def test_multiple_statements_rejected():
    with pytest.raises(GuardrailViolation, match="multiple"):
        validate_sql("SELECT 1; DROP TABLE patients;")


# ---------- 2. PHI projection ----------

@pytest.mark.parametrize("sql", [
    "SELECT first_name FROM patients",
    "SELECT p.last_name, COUNT(*) FROM patients p GROUP BY 1",
    "SELECT mrn, encounter_id FROM encounters JOIN patients USING(patient_id)",
])
def test_phi_columns_blocked(sql):
    with pytest.raises(GuardrailViolation, match="PHI"):
        validate_sql(sql)


def test_phi_column_in_where_allowed():
    """PHI in WHERE/JOIN is fine — only projection leaks identity."""
    validate_sql("SELECT COUNT(*) FROM patients WHERE last_name = 'Smith'")


def test_star_select_blocked_on_phi_tables():
    """SELECT * from a PHI-containing table would leak PHI."""
    with pytest.raises(GuardrailViolation, match="PHI|SELECT \\*"):
        validate_sql("SELECT * FROM patients")


def test_star_select_allowed_on_non_phi_tables():
    validate_sql("SELECT * FROM readmissions")


# ---------- 3. Unknown table ----------

def test_unknown_table_rejected():
    with pytest.raises(GuardrailViolation, match="unknown table"):
        validate_sql("SELECT * FROM nonexistent_table")


# ---------- 4. Row limit injection ----------

def test_inject_limit_when_missing():
    out = enforce_row_limit("SELECT * FROM readmissions", max_rows=100)
    assert "LIMIT 100" in out.upper()


def test_preserve_existing_smaller_limit():
    out = enforce_row_limit("SELECT * FROM readmissions LIMIT 10", max_rows=100)
    assert "LIMIT 10" in out


def test_clamp_larger_limit():
    out = enforce_row_limit("SELECT * FROM readmissions LIMIT 5000", max_rows=100)
    assert "LIMIT 100" in out.upper()
    assert "5000" not in out


# ---------- 5. Aggregate queries don't need LIMIT shrunk to absurd ----------

def test_count_query_still_gets_limit():
    out = enforce_row_limit("SELECT COUNT(*) FROM readmissions", max_rows=100)
    assert "LIMIT" in out.upper()
