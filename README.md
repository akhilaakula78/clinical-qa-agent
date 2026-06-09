# Clinical Data Q&A Agent

A multi-agent system that answers natural-language questions about hospital
operational data. Ask questions in plain English — the system plans, generates
SQL, validates it for safety, executes against the EHR database, and returns
a clear answer.

## Architecture

```
                ┌──────────────┐
   question ──▶ │   Planner    │ ──▶ clarify / reject
                └──────┬───────┘
                       │ sql
                       ▼
                ┌──────────────┐
                │ SQL Generator│ ◀────────┐
                └──────┬───────┘          │
                       ▼                  │ retry (≤3)
                ┌──────────────┐  fail    │
                │  Validator   │ ─────────┘
                │  (guardrails)│
                └──────┬───────┘
                       │ pass + safe_sql
                       ▼
                ┌──────────────┐    ┌──────────────┐
                │   Executor   │──▶ │   Answerer   │ ──▶ NL answer
                │ (DuckDB / SF)│    │  (Gemini)    │
                └──────────────┘    └──────────────┘
```

Orchestrated as a LangGraph `StateGraph`. Every transition emits a JSONL
audit event (session id, step, detail).

## Components

| Component | Tech | Notes |
|---|---|---|
| Planner | Gemini + Pydantic structured output | Routes question to `sql` / `clarify` / `reject`. |
| SQL Generator | Gemini + Pydantic | Schema-aware. Retries with prior error context. |
| Validator | `sqlglot` AST | SELECT-only, PHI projection block, unknown table block, row-limit injection. No LLM — deterministic. |
| Executor | DuckDB (default) or Snowflake | `DBAdapter` Protocol, env-switchable. |
| Answerer | Gemini | Summarises result rows in 1–3 sentences. |
| Orchestrator | LangGraph `StateGraph` | Conditional edges, ≤3 retry loop, audit. |
| Audit | JSONL | One line per agent step. Session-scoped. |
| API | FastAPI | `POST /ask` returns answer + SQL + rows + audit. |
| UI | Streamlit | Chat-style, shows generated SQL, results, audit trail. |

## Data

`data/generate_synthetic.py` produces a synthetic oncology EHR — no real PHI:

| Table | Rows (500-pt run) | Notes |
|---|---|---|
| patients | 500 | PHI columns flagged. |
| providers | 25 | Oncology / Hematology / Rad Onc / Surgical Onc / Palliative. |
| encounters | ~2,200 | Inpatient / Outpatient / ED / Observation. |
| diagnoses | ~5,500 | ICD-10. C-codes for cancer. |
| medications | ~4,500 | Chemo + supportive care. |
| lab_results | ~7,700 | CBC + chem panel. |
| readmissions | ~470 | Precomputed 30-day flag. |

## Guardrails

AST-based SQL safety layer using `sqlglot` — runs before every execution:

1. **SELECT-only.** `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `CREATE`, `GRANT` → reject.
2. **Single statement.** Multi-statement input → reject.
3. **No PHI projection.** `first_name`, `last_name`, `mrn` blocked in SELECT (allowed in WHERE/JOIN).
4. **SELECT \* gated.** Forbidden on tables containing PHI (e.g. `patients`).
5. **Unknown tables.** Anything not in `schema.yaml` → reject.
6. **Row limit.** `LIMIT` injected when missing; clamped when above `MAX_ROWS`.

The validator returns a `safe_sql` string — the executor never runs raw model output.

## Quickstart

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Generate synthetic data
python data/generate_synthetic.py --patients 500

# 3. Configure
cp .env.example .env
# edit .env: set GOOGLE_API_KEY and GEMINI_MODEL=gemini-2.5-flash

# 4. Tests
pytest

# 5. Run API + UI (two terminals)
uvicorn clinical_qa.api.main:app --reload
streamlit run src/clinical_qa/ui/app.py
```

## Sample questions

- "What is the overall 30-day readmission rate?"
- "Top 5 primary cancer diagnoses by encounter count"
- "Average length of stay for inpatient encounters by department"
- "How many patients received Cisplatin in 2024?"
- "Readmission rate for breast cancer patients (ICD-10 starting with C50)"
- "Percentage of WBC labs that are abnormally low"

Blocked by guardrails:

- "Show me first_name and last_name of all patients" — PHI projection blocked.
- "DROP TABLE patients" — non-SELECT blocked.
- "SELECT * FROM patients" — SELECT * on PHI table blocked.

## Switching to Snowflake

```bash
pip install -e ".[snowflake]"
# .env:
DB_BACKEND=snowflake
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_DATABASE=...
SNOWFLAKE_SCHEMA=...
```

The `DBAdapter` Protocol means agents and guardrails are engine-agnostic.

## Audit log

Every request appends JSONL lines to `audit.jsonl`:

```json
{"ts":"2026-06-09T...","session_id":"...","step":"planner","detail":{"action":"sql","rationale":"..."}}
{"ts":"...","step":"sql_generator","detail":{"sql":"SELECT ...","attempt":1}}
{"ts":"...","step":"validator","detail":{"ok":true,"safe_sql":"SELECT ... LIMIT 1000"}}
{"ts":"...","step":"executor","detail":{"row_count":1,"columns":["rate"]}}
{"ts":"...","step":"answerer","detail":{"answer_preview":"..."}}
```

## Tests

33 tests covering guardrails (21), agent contracts (8), and end-to-end graph
behavior (4 — including retry-on-violation loop and reject-after-max-retries).

```bash
pytest -v
```

## Layout

```
src/clinical_qa/
  config.py              # pydantic-settings, env-driven
  db/
    adapter.py           # DBAdapter Protocol + factory
    duckdb_adapter.py
    snowflake_adapter.py
  schema/
    schema.yaml          # table/column catalog for LLM prompt
    phi_columns.yaml     # PHI block-list
  guardrails/
    sql_safety.py        # AST checks + row-limit enforcement
  agents/
    planner.py           # route: sql | clarify | reject
    sql_generator.py     # NL → SQL with retry context
    validator.py         # guardrail wrapper, no LLM
    answerer.py          # rows → NL summary
    llm.py               # LLM client (Gemini / Anthropic, env-switchable)
  graph/
    state.py             # AgentState TypedDict
    orchestrator.py      # StateGraph wiring
  audit/
    logger.py            # JSONL per-step audit
  api/main.py            # FastAPI /ask
  ui/app.py              # Streamlit chat
```

## Stack

Python 3.9+ · LangGraph · Gemini (default) / Anthropic · DuckDB · sqlglot · FastAPI · Streamlit
