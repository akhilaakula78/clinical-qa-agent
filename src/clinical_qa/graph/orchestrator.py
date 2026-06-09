"""LangGraph StateGraph wiring the 3 agents + executor + answerer.

Flow:
  question -> planner -> [sql | clarify | reject]
    sql path: sqlgen -> validator -> [execute | retry sqlgen (max 2)]
              execute -> answerer -> END
    clarify path: emit clarification -> END
    reject path: emit reject reason -> END
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from clinical_qa.agents.answerer import compose_answer
from clinical_qa.agents.planner import plan as run_planner
from clinical_qa.agents.sql_generator import generate_sql
from clinical_qa.agents.validator import validate
from clinical_qa.audit.logger import log_event
from clinical_qa.db.adapter import get_adapter
from clinical_qa.graph.state import AgentState

MAX_SQL_ATTEMPTS = 3


def _audit(state: AgentState, step: str, detail: dict[str, Any]) -> list:
    events = list(state.get("audit", []))
    events.append({"step": step, "detail": detail})
    log_event(step, detail)
    return events


def planner_node(state: AgentState) -> dict:
    p = run_planner(state["question"])
    return {
        "plan_action": p.action,
        "plan_rationale": p.rationale,
        "clarification": p.clarification,
        "attempts": 0,
        "audit": _audit(state, "planner", {
            "action": p.action, "rationale": p.rationale,
            "clarification": p.clarification,
        }),
    }


def sqlgen_node(state: AgentState) -> dict:
    attempts = state.get("attempts", 0)
    g = generate_sql(
        state["question"],
        previous_error=state.get("validation_error") if attempts > 0 else None,
        previous_sql=state.get("sql") if attempts > 0 else None,
    )
    return {
        "sql": g.sql,
        "attempts": attempts + 1,
        "audit": _audit(state, "sql_generator", {
            "sql": g.sql, "explanation": g.explanation, "attempt": attempts + 1,
        }),
    }


def validator_node(state: AgentState) -> dict:
    r = validate(state["sql"])
    return {
        "sql_safe": r.safe_sql,
        "validation_error": r.error,
        "audit": _audit(state, "validator", {
            "ok": r.ok, "error": r.error, "safe_sql": r.safe_sql,
        }),
    }


def executor_node(state: AgentState) -> dict:
    adapter = get_adapter()
    try:
        res = adapter.execute(state["sql_safe"])
        return {
            "query_columns": res.columns,
            "query_rows": res.rows,
            "row_count": res.row_count,
            "audit": _audit(state, "executor", {
                "row_count": res.row_count, "columns": res.columns,
            }),
        }
    except Exception as e:
        return {
            "error": f"execution error: {e}",
            "audit": _audit(state, "executor", {"error": str(e)}),
        }
    finally:
        adapter.close()


def answerer_node(state: AgentState) -> dict:
    answer = compose_answer(
        state["question"], state["sql_safe"],
        state["query_columns"], state["query_rows"],
    )
    return {
        "final_answer": answer,
        "audit": _audit(state, "answerer", {"answer_preview": answer[:200]}),
    }


def clarify_node(state: AgentState) -> dict:
    msg = state.get("clarification") or "Could you clarify your question?"
    return {"final_answer": f"[clarification needed] {msg}"}


def reject_node(state: AgentState) -> dict:
    return {"final_answer": f"[rejected] {state.get('plan_rationale', 'out of scope')}"}


def route_after_planner(state: AgentState) -> str:
    return {"sql": "sqlgen", "clarify": "clarify", "reject": "reject"}[state["plan_action"]]


def route_after_validator(state: AgentState) -> str:
    if state.get("validation_error") is None:
        return "executor"
    if state.get("attempts", 0) >= MAX_SQL_ATTEMPTS:
        return "reject_after_retries"
    return "sqlgen"


def reject_after_retries_node(state: AgentState) -> dict:
    return {"final_answer": (
        f"[rejected] could not generate safe SQL after {MAX_SQL_ATTEMPTS} attempts. "
        f"Last error: {state.get('validation_error')}"
    )}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("sqlgen", sqlgen_node)
    g.add_node("validator", validator_node)
    g.add_node("executor", executor_node)
    g.add_node("answerer", answerer_node)
    g.add_node("clarify", clarify_node)
    g.add_node("reject", reject_node)
    g.add_node("reject_after_retries", reject_after_retries_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, {
        "sqlgen": "sqlgen", "clarify": "clarify", "reject": "reject",
    })
    g.add_edge("sqlgen", "validator")
    g.add_conditional_edges("validator", route_after_validator, {
        "executor": "executor",
        "sqlgen": "sqlgen",
        "reject_after_retries": "reject_after_retries",
    })
    g.add_edge("executor", "answerer")
    g.add_edge("answerer", END)
    g.add_edge("clarify", END)
    g.add_edge("reject", END)
    g.add_edge("reject_after_retries", END)

    return g.compile()


def run(question: str) -> dict:
    graph = build_graph()
    return graph.invoke({"question": question, "audit": []})
