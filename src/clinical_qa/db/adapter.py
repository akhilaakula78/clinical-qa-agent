from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    @property
    def row_count(self) -> int:
        return len(self.rows)


class DBAdapter(Protocol):
    dialect: str

    def execute(self, sql: str) -> QueryResult: ...
    def list_tables(self) -> list[str]: ...
    def describe(self, table: str) -> list[tuple[str, str]]: ...
    def close(self) -> None: ...


def get_adapter(backend: str | None = None) -> DBAdapter:
    from clinical_qa.config import settings

    backend = backend or settings.db_backend
    if backend == "duckdb":
        from clinical_qa.db.duckdb_adapter import DuckDBAdapter
        return DuckDBAdapter(settings.duckdb_path)
    if backend == "snowflake":
        from clinical_qa.db.snowflake_adapter import SnowflakeAdapter
        return SnowflakeAdapter()
    raise ValueError(f"Unknown backend: {backend}")
