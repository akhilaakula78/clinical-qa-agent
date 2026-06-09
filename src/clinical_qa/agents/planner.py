"""Planner agent. Routes question → sql | clarify | reject."""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from clinical_qa.agents.llm import get_llm
from clinical_qa.schema import schema_prompt


class Plan(BaseModel):
    action: Literal["sql", "clarify", "reject"] = Field(
        description="sql=answerable via SQL; clarify=ambiguous; reject=out of scope or unsafe"
    )
    rationale: str = Field(description="One sentence: why this action.")
    clarification: str | None = Field(
        default=None, description="If action=clarify, the question to ask the user."
    )


SYSTEM = """You are the Query Planner for a hospital analytics agent.

Decide how to handle the user's question:

  - "sql": question can be answered by querying the EHR schema below.
  - "clarify": question is ambiguous (vague time range, ambiguous cohort, undefined metric) — ask one focused follow-up.
  - "reject": out of scope (clinical advice, individual PHI lookup by name, non-data questions) or unsafe.

Reject requests that name individual patients or ask for identifying info.
Prefer "sql" when the question is well-formed and answerable from the schema.

SCHEMA:
{schema}
"""


def plan(question: str) -> Plan:
    llm = get_llm()
    structured = llm.with_structured_output(Plan)
    msgs = [
        SystemMessage(content=SYSTEM.format(schema=schema_prompt())),
        HumanMessage(content=question),
    ]
    return structured.invoke(msgs)
