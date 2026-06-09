"""SQL safety guardrails. AST-based — no regex hacks.

Policy:
  - Only SELECT (incl. CTEs / WITH) is allowed.
  - Exactly one top-level statement.
  - No PHI columns in SELECT projection (WHERE/JOIN OK).
  - SELECT * forbidden on tables that contain PHI.
  - Every referenced table must exist in the schema catalog.
  - Row limit enforced (injected or clamped).
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from clinical_qa.schema import load_phi, load_schema


class GuardrailViolation(Exception):
    pass


_DESTRUCTIVE_NODES = (
    exp.Delete, exp.Insert, exp.Update, exp.Drop, exp.Alter,
    exp.Create, exp.Command, exp.TruncateTable, exp.Grant,
)


def _parse(sql: str, dialect: str = "duckdb") -> list[exp.Expression]:
    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        raise GuardrailViolation(f"parse error: {e}") from e
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        raise GuardrailViolation("empty statement")
    return parsed


def _known_tables() -> set[str]:
    return set(load_schema()["tables"].keys())


def _phi_tables() -> set[str]:
    return set(load_phi()["phi_columns"].keys())


def _forbidden_cols() -> set[str]:
    return set(load_phi()["forbidden_in_select"])


def validate_sql(sql: str, dialect: str = "duckdb") -> None:
    """Raise GuardrailViolation if SQL violates policy. Return None if OK."""
    parsed = _parse(sql, dialect)

    if len(parsed) > 1:
        raise GuardrailViolation("multiple statements not allowed")

    tree = parsed[0]

    if isinstance(tree, _DESTRUCTIVE_NODES):
        raise GuardrailViolation(f"destructive / non-SELECT statement: {type(tree).__name__}")
    if not isinstance(tree, (exp.Select, exp.Union, exp.With, exp.Subquery)):
        raise GuardrailViolation(f"only SELECT allowed, got: {type(tree).__name__}")

    for node in tree.walk():
        if isinstance(node, _DESTRUCTIVE_NODES):
            raise GuardrailViolation(f"non-SELECT subnode: {type(node).__name__}")

    known = _known_tables()
    referenced = set()
    for t in tree.find_all(exp.Table):
        name = (t.name or "").lower()
        if name:
            referenced.add(name)
    unknown = referenced - known
    if unknown:
        raise GuardrailViolation(f"unknown table(s): {sorted(unknown)}")

    phi_tables_in_query = referenced & _phi_tables()
    forbidden = _forbidden_cols()

    for select in tree.find_all(exp.Select):
        for proj in select.expressions:
            if isinstance(proj, exp.Star) or (
                isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star)
            ):
                if phi_tables_in_query:
                    raise GuardrailViolation(
                        f"SELECT * forbidden — query touches PHI table(s): "
                        f"{sorted(phi_tables_in_query)}"
                    )
                continue
            for col in proj.find_all(exp.Column):
                col_name = (col.name or "").lower()
                if col_name in forbidden:
                    raise GuardrailViolation(f"PHI column '{col_name}' in SELECT projection")


def enforce_row_limit(sql: str, max_rows: int, dialect: str = "duckdb") -> str:
    """Inject LIMIT, or clamp existing LIMIT down to max_rows."""
    parsed = _parse(sql, dialect)
    if len(parsed) > 1:
        raise GuardrailViolation("multiple statements not allowed")
    tree = parsed[0]

    target = tree
    if isinstance(tree, exp.With):
        target = tree.this
    if not isinstance(target, (exp.Select, exp.Union, exp.Subquery)):
        return sql

    existing = target.args.get("limit")
    if existing is not None:
        try:
            n = int(existing.expression.name)
            if n > max_rows:
                target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        except (AttributeError, ValueError):
            target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))

    return tree.sql(dialect=dialect)
