"""SQL Generator agent. NL question → DuckDB/Snowflake SQL."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from clinical_qa.agents.llm import get_llm
from clinical_qa.schema import schema_prompt


class GeneratedSQL(BaseModel):
    sql: str = Field(description="Single SELECT statement. No trailing semicolon.")
    explanation: str = Field(description="One sentence describing what the query computes.")


SYSTEM = """You generate read-only SQL for a hospital analytics agent.

RULES (the validator will reject violations):
  - SELECT only. No DDL/DML/multiple statements.
  - Never project PHI columns: first_name, last_name, mrn. They may appear in WHERE only.
  - Avoid SELECT *. List columns explicitly. SELECT * is allowed only on non-PHI tables.
  - Use only tables/columns from the schema below.
  - Dialect: {dialect} (DuckDB-compatible / Snowflake-compatible standard SQL).
  - Prefer aggregates over raw rows when the question asks for a rate/count/avg.
  - Use the readmissions table for 30-day readmission questions (precomputed).
  - For "this quarter" / "last month" relative dates, use CURRENT_DATE arithmetic.
  - Return only the SQL — no ```sql fences.

SCHEMA:
{schema}

{retry_context}
"""


def generate_sql(
    question: str,
    dialect: str = "duckdb",
    previous_error: str | None = None,
    previous_sql: str | None = None,
) -> GeneratedSQL:
    llm = get_llm()
    structured = llm.with_structured_output(GeneratedSQL)
    retry_context = ""
    if previous_error and previous_sql:
        retry_context = (
            f"PREVIOUS ATTEMPT FAILED:\nSQL: {previous_sql}\n"
            f"ERROR: {previous_error}\nFix the issue.\n"
        )
    msgs = [
        SystemMessage(content=SYSTEM.format(
            schema=schema_prompt(), dialect=dialect, retry_context=retry_context,
        )),
        HumanMessage(content=question),
    ]
    return structured.invoke(msgs)
