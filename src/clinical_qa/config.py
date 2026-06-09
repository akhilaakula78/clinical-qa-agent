from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["anthropic", "gemini"] = "gemini"

    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-7"

    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    db_backend: Literal["duckdb", "snowflake"] = "duckdb"
    duckdb_path: str = "./data/clinical.duckdb"

    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_password: str = ""
    snowflake_warehouse: str = ""
    snowflake_database: str = ""
    snowflake_schema: str = ""

    max_rows: int = 1000
    audit_log_path: str = "./audit.jsonl"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]


settings = Settings()
