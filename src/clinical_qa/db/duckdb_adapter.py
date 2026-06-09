from __future__ import annotations

import duckdb

from clinical_qa.db.adapter import QueryResult


class DuckDBAdapter:
    dialect = "duckdb"

    def __init__(self, path: str):
        self._con = duckdb.connect(path, read_only=True)

    def execute(self, sql: str) -> QueryResult:
        cur = self._con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return QueryResult(columns=cols, rows=[list(r) for r in rows])

    def list_tables(self) -> list[str]:
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        return [r[0] for r in rows]

    def describe(self, table: str) -> list[tuple[str, str]]:
        rows = self._con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def close(self) -> None:
        self._con.close()
