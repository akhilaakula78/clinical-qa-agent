"""End-to-end graph tests with mocked LLM. Uses real DuckDB for execution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from clinical_qa.agents.planner import Plan
from clinical_qa.agents.sql_generator import GeneratedSQL


def _mock_llm(planner_action="sql", sql="SELECT COUNT(*) AS n FROM readmissions",
              clarification=None, answer_text="The result is 469."):
    """Patches all three LLM hooks consistently."""
    planner_fake = MagicMock()
    planner_fake.with_structured_output.return_value.invoke.return_value = Plan(
        action=planner_action, rationale="test", clarification=clarification,
    )
    sqlgen_fake = MagicMock()
    sqlgen_fake.with_structured_output.return_value.invoke.return_value = GeneratedSQL(
        sql=sql, explanation="test",
    )
    answerer_fake = MagicMock()
    answerer_fake.invoke.return_value = MagicMock(content=answer_text)
    return planner_fake, sqlgen_fake, answerer_fake


def test_e2e_sql_path():
    planner_llm, sqlgen_llm, answerer_llm = _mock_llm()
    with patch("clinical_qa.agents.planner.get_llm", return_value=planner_llm), \
         patch("clinical_qa.agents.sql_generator.get_llm", return_value=sqlgen_llm), \
         patch("clinical_qa.agents.answerer.get_llm", return_value=answerer_llm):
        from clinical_qa.graph.orchestrator import run
        state = run("How many readmission records exist?")

    assert state["plan_action"] == "sql"
    assert state["query_columns"] == ["n"]
    assert state["row_count"] == 1
    assert state["query_rows"][0][0] == 469
    assert "469" in state["final_answer"]
    audit_steps = [e["step"] for e in state["audit"]]
    assert "planner" in audit_steps
    assert "sql_generator" in audit_steps
    assert "validator" in audit_steps
    assert "executor" in audit_steps


def test_e2e_clarify_path():
    planner_llm, sqlgen_llm, answerer_llm = _mock_llm(
        planner_action="clarify", clarification="Which quarter?",
    )
    with patch("clinical_qa.agents.planner.get_llm", return_value=planner_llm), \
         patch("clinical_qa.agents.sql_generator.get_llm", return_value=sqlgen_llm), \
         patch("clinical_qa.agents.answerer.get_llm", return_value=answerer_llm):
        from clinical_qa.graph.orchestrator import run
        state = run("How many recently?")

    assert state["plan_action"] == "clarify"
    assert "Which quarter?" in state["final_answer"]


def test_e2e_retry_on_phi_violation():
    """First SQL has PHI; second attempt is safe; should succeed via retry loop."""
    planner_fake = MagicMock()
    planner_fake.with_structured_output.return_value.invoke.return_value = Plan(
        action="sql", rationale="test",
    )
    sqlgen_fake = MagicMock()
    sqlgen_fake.with_structured_output.return_value.invoke.side_effect = [
        GeneratedSQL(sql="SELECT first_name FROM patients", explanation="bad"),
        GeneratedSQL(sql="SELECT COUNT(*) AS n FROM patients", explanation="good"),
    ]
    answerer_fake = MagicMock()
    answerer_fake.invoke.return_value = MagicMock(content="500 patients.")

    with patch("clinical_qa.agents.planner.get_llm", return_value=planner_fake), \
         patch("clinical_qa.agents.sql_generator.get_llm", return_value=sqlgen_fake), \
         patch("clinical_qa.agents.answerer.get_llm", return_value=answerer_fake):
        from clinical_qa.graph.orchestrator import run
        state = run("How many patients?")

    assert state["attempts"] == 2
    assert state["row_count"] == 1
    assert state["query_rows"][0][0] == 500


def test_e2e_max_retries_rejects():
    """Persistent PHI violation should reject after MAX_SQL_ATTEMPTS."""
    planner_fake = MagicMock()
    planner_fake.with_structured_output.return_value.invoke.return_value = Plan(
        action="sql", rationale="test",
    )
    sqlgen_fake = MagicMock()
    sqlgen_fake.with_structured_output.return_value.invoke.return_value = GeneratedSQL(
        sql="SELECT first_name FROM patients", explanation="bad",
    )
    answerer_fake = MagicMock()

    with patch("clinical_qa.agents.planner.get_llm", return_value=planner_fake), \
         patch("clinical_qa.agents.sql_generator.get_llm", return_value=sqlgen_fake), \
         patch("clinical_qa.agents.answerer.get_llm", return_value=answerer_fake):
        from clinical_qa.graph.orchestrator import run
        state = run("Give me patient names")

    assert "rejected" in state["final_answer"].lower()
    assert state["attempts"] == 3
