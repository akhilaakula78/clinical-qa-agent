"""Validator agent. Applies guardrails + row limit. Pure code, no LLM."""
from __future__ import annotations

from dataclasses import dataclass

from clinical_qa.config import settings
from clinical_qa.guardrails.sql_safety import (
    GuardrailViolation,
    enforce_row_limit,
    validate_sql,
)


@dataclass
class ValidationResult:
    ok: bool
    safe_sql: str | None
    error: str | None


def validate(sql: str, dialect: str = "duckdb") -> ValidationResult:
    try:
        validate_sql(sql, dialect=dialect)
        safe = enforce_row_limit(sql, max_rows=settings.max_rows, dialect=dialect)
        return ValidationResult(ok=True, safe_sql=safe, error=None)
    except GuardrailViolation as e:
        return ValidationResult(ok=False, safe_sql=None, error=str(e))
