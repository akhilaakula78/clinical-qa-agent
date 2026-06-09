from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class AuditEvent(TypedDict):
    step: str
    detail: dict[str, Any]


class AgentState(TypedDict, total=False):
    question: str
    plan_action: Literal["sql", "clarify", "reject"]
    plan_rationale: str
    clarification: Optional[str]
    sql: Optional[str]
    sql_safe: Optional[str]
    validation_error: Optional[str]
    query_columns: Optional[list[str]]
    query_rows: Optional[list[list[Any]]]
    row_count: Optional[int]
    attempts: int
    final_answer: Optional[str]
    error: Optional[str]
    audit: list[AuditEvent]
