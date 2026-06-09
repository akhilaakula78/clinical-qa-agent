"""Streamlit UI. Talks to the FastAPI /ask endpoint."""
from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("CLINICAL_QA_API", "http://localhost:8000")

st.set_page_config(page_title="Clinical Q&A Agent", layout="wide")
st.title("Clinical Q&A Agent")
st.caption("Multi-agent SQL Q&A over synthetic oncology EHR data. "
           "LangGraph + Claude. PHI guardrails + 30-day readmissions precomputed.")

with st.sidebar:
    st.header("Example questions")
    examples = [
        "What is the overall 30-day readmission rate?",
        "Top 5 primary cancer diagnoses by encounter count",
        "Average length of stay for inpatient encounters by department",
        "How many patients received Cisplatin in 2024?",
        "Readmission rate for breast cancer patients (ICD-10 starting with C50)",
        "Percentage of WBC labs that are abnormally low",
    ]
    for q in examples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["pending"] = q

question = st.text_input(
    "Ask a question:",
    value=st.session_state.get("pending", ""),
    placeholder="e.g., What is the 30-day readmission rate for CHF patients this quarter?",
)
st.session_state["pending"] = ""

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Planning → generating SQL → validating → executing…"):
        try:
            r = httpx.post(f"{API}/ask", json={"question": question}, timeout=120.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(data["answer"])

    cols = st.columns(3)
    cols[0].metric("Plan", data["plan_action"])
    cols[1].metric("Attempts", data["attempts"])
    cols[2].metric("Rows", data["row_count"] or 0)

    if data.get("sql"):
        st.subheader("Generated SQL")
        st.code(data["sql"], language="sql")
    if data.get("sql_safe") and data["sql_safe"] != data.get("sql"):
        st.subheader("After guardrails (row limit injected)")
        st.code(data["sql_safe"], language="sql")
    if data.get("rows"):
        st.subheader("Result")
        df = pd.DataFrame(data["rows"], columns=data["columns"])
        st.dataframe(df, use_container_width=True)

    with st.expander("Audit trail (every agent step)"):
        st.code(json.dumps(data["audit"], indent=2, default=str), language="json")
