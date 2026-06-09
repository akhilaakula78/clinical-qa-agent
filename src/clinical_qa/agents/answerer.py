"""Final answer composer. Turns query results into NL response."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from clinical_qa.agents.llm import get_llm

SYSTEM = """You summarize SQL query results for a clinical analyst.

  - State the answer in 1-3 sentences.
  - Quote the numbers from the result table exactly.
  - Mention the cohort/timeframe the query used so the user can verify intent.
  - Do not invent numbers not present in the result.
  - Do not give clinical advice.
"""


def compose_answer(question: str, sql: str, columns: list[str], rows: list[list]) -> str:
    if not rows:
        return "Query returned no rows."
    preview = [dict(zip(columns, r)) for r in rows[:20]]
    llm = get_llm()
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Question: {question}\n\nSQL:\n{sql}\n\n"
            f"Result columns: {columns}\nRows (up to 20): {preview}\n"
            f"Total rows returned: {len(rows)}"
        )),
    ]
    return llm.invoke(msgs).content
