from __future__ import annotations

from clinical_qa.config import settings
from clinical_qa.db.adapter import QueryResult


class SnowflakeAdapter:
    """Thin Snowflake adapter.

    Requires the `snowflake` extra: `pip install -e ".[snowflake]"`.
    Reads credentials from settings (env-driven).
    """

    dialect = "snowflake"

    def __init__(self):
        try:
            import snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python not installed. "
                'Install with: pip install -e ".[snowflake]"'
            ) from e
        self._con = snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            password=settings.snowflake_password,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
        )

    def execute(self, sql: str) -> QueryResult:
        cur = self._con.cursor()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return QueryResult(columns=cols, rows=[list(r) for r in rows])
        finally:
            cur.close()

    def list_tables(self) -> list[str]:
        r = self.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{settings.snowflake_schema}'"
        )
        return [row[0] for row in r.rows]

    def describe(self, table: str) -> list[tuple[str, str]]:
        r = self.execute(
            "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_NAME = '{table.upper()}' "
            f"AND TABLE_SCHEMA = '{settings.snowflake_schema}' "
            "ORDER BY ORDINAL_POSITION"
        )
        return [(row[0], row[1]) for row in r.rows]

    def close(self) -> None:
        self._con.close()
