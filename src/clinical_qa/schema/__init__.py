from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(_DIR / "schema.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_phi() -> dict:
    with open(_DIR / "phi_columns.yaml") as f:
        return yaml.safe_load(f)


def schema_prompt() -> str:
    """Render schema as compact text for LLM prompt context."""
    s = load_schema()
    lines = [f"Database: {s['database']}", s["description"].strip(), "", "TABLES:"]
    for tname, tdef in s["tables"].items():
        lines.append(f"\n{tname} — {tdef['description']}")
        for c in tdef["columns"]:
            desc = f"  -- {c['description']}" if c.get("description") else ""
            lines.append(f"  {c['name']} {c['type']}{desc}")
    lines.append("\nCOMMON JOINS:")
    for j in s["common_joins"]:
        lines.append(f"  {j}")
    return "\n".join(lines)
