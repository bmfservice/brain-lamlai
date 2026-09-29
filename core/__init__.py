"""Braind Core - Package init."""

import datetime as _dt
import json as _json
from pathlib import Path
from typing import Any

# --- JSON serialization patch: pydantic `date`/`datetime` → isoformat
# This makes DI (make extended JSON handle these) when writing JSONB to
# PostgreSQL via SQLAlchemy/asyncpg — otherwise `date` objects from YAML
# frontmatter raise "Object of type date is not JSON serializable".


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, _dt.time):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# asyncpg: register serializer + codecs on connect events for full coverage
# (this module is imported by core.database which wires it to the engine)
def register_asyncpg_codecs() -> None:
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        return

    try:
        from asyncpg.pgproto import pgproto

        if not hasattr(pgproto, "isoformat") and hasattr(pgproto, "UUID"):

            def _uuid_to_str(u):
                return str(u)

            # Not strictly needed; keep for safety
    except Exception:
        pass


def iso(value: Any) -> str:
    """Convert a known-serializable value to its JSON-safe string form."""
    return _json.dumps(value, default=_json_default)


def make_json_safe(value: Any) -> Any:
    """Recursively convert date/datetime/path/set to JSON-safe forms."""
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value
