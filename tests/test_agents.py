"""Agent tests. Validator is pure code; planner/sqlgen use mocked LLM."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clinical_qa.agents.planner import Plan, plan
from clinical_qa.agents.sql_generator import GeneratedSQL, generate_sql
from clinical_qa.agents.validator import validate


# ---------- Validator (no LLM) ----------

def test_validator_passes_safe_sql():
    r = validate("SELECT COUNT(*) FROM readmissions")
    assert r.ok
    assert "LIMIT" in r.safe_sql.upper()
    assert r.error is None


def test_validator_blocks_destructive():
    r = validate("DROP TABLE patients")
    assert not r.ok
    assert "destructive" in r.error.lower() or "non-SELECT" in r.error


def test_validator_blocks_phi_projection():
    r = validate("SELECT first_name FROM patients")
    assert not r.ok
    assert "PHI" in r.error


def test_validator_clamps_row_limit():
    r = validate("SELECT * FROM readmissions LIMIT 99999")
    assert r.ok
    assert "99999" not in r.safe_sql


# ---------- Planner (mocked LLM) ----------

@patch("clinical_qa.agents.planner.get_llm")
def test_planner_routes_to_sql(mock_llm):
    fake = MagicMock()
    fake.with_structured_output.return_value.invoke.return_value = Plan(
        action="sql", rationale="answerable from schema"
    )
    mock_llm.return_value = fake

    p = plan("What is the 30-day readmission rate?")
    assert p.action == "sql"


@patch("clinical_qa.agents.planner.get_llm")
def test_planner_clarifies_ambiguous(mock_llm):
    fake = MagicMock()
    fake.with_structured_output.return_value.invoke.return_value = Plan(
        action="clarify",
        rationale="time range is unclear",
        clarification="Which quarter?",
    )
    mock_llm.return_value = fake

    p = plan("How many readmissions recently?")
    assert p.action == "clarify"
    assert p.clarification == "Which quarter?"


# ---------- SQL Generator (mocked LLM) ----------

@patch("clinical_qa.agents.sql_generator.get_llm")
def test_sqlgen_returns_select(mock_llm):
    fake = MagicMock()
    fake.with_structured_output.return_value.invoke.return_value = GeneratedSQL(
        sql="SELECT AVG(CAST(readmitted_30d AS INTEGER)) FROM readmissions",
        explanation="Mean readmission rate.",
    )
    mock_llm.return_value = fake

    g = generate_sql("What is the 30-day readmission rate?")
    assert g.sql.upper().startswith("SELECT")


@patch("clinical_qa.agents.sql_generator.get_llm")
def test_sqlgen_retry_context_passed(mock_llm):
    """When retrying after a validation error, context is included in prompt."""
    fake = MagicMock()
    structured = fake.with_structured_output.return_value
    structured.invoke.return_value = GeneratedSQL(sql="SELECT 1", explanation="x")
    mock_llm.return_value = fake

    generate_sql(
        "q", previous_error="PHI column 'first_name' in SELECT projection",
        previous_sql="SELECT first_name FROM patients",
    )
    sent_messages = structured.invoke.call_args[0][0]
    system = sent_messages[0].content
    assert "PREVIOUS ATTEMPT FAILED" in system
    assert "first_name" in system
