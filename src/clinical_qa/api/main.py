"""FastAPI service exposing the agent."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from clinical_qa.graph.orchestrator import run

app = FastAPI(title="Clinical Q&A Agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    plan_action: str
    sql: str | None = None
    sql_safe: str | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    attempts: int
    audit: list[dict]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="empty question")
    state = run(req.question)
    return AskResponse(
        answer=state.get("final_answer", ""),
        plan_action=state.get("plan_action", ""),
        sql=state.get("sql"),
        sql_safe=state.get("sql_safe"),
        columns=state.get("query_columns"),
        rows=state.get("query_rows"),
        row_count=state.get("row_count"),
        attempts=state.get("attempts", 0),
        audit=state.get("audit", []),
    )
