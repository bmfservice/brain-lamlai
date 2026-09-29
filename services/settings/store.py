"""Settings store — DB-backed read/write (Brain Second Phase 1a).

The single write path for setting values. Every ``set_value``:
  1. loads the key's definition (an unknown key is refused — the store never
     invents settings),
  2. checks the value against the System Hard Rules (``hard_rules``),
  3. validates + coerces it against the typed schema (``schema.validate``),
  4. appends a ``setting_history`` row carrying the REAL previous value,
  5. upserts the current ``setting_values`` row,
  6. writes an audit row (reusing the governance ``PolicyEngine.audit`` so the
     settings trail lands in the same ``audit_logs`` table).

There is no silent change: a value only reaches the DB after it passes
validation, and every write leaves a versioned, attributable history row.

Reads (``get_value``/``list_values``) return the unwrapped value; the JSONB
``{"v": ...}`` wrapping is an internal storage detail (see ``schema._wrap``).

The seven-scope priority *resolution* is Phase 1b; this store deals in one
explicit (key, scope, scope_ref) at a time and does not yet choose between them.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core import database as db
from core.logging import get_logger
from services.settings import hard_rules
from services.settings.schema import (
    SettingScope,
    SettingSpec,
    SettingValueError,
    _wrap,
    validate,
)

logger = get_logger(__name__)


class SettingsStore:
    """Typed, validated, versioned access to setting definitions and values."""

    def __init__(self, session_factory, *, auditor=None) -> None:
        self.session_factory = session_factory
        # Optional: a PolicyEngine (or anything with .audit(...)) to record
        # changes into audit_logs. None → audit is skipped (tests, early boot).
        self.auditor = auditor

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    async def upsert_definition(self, spec: SettingSpec) -> None:
        """Insert or update one definition row from a ``SettingSpec``."""
        row = spec.to_row()
        key = row.pop("key")
        async with self.session_factory() as session:
            await session.execute(
                pg_insert(db.SettingDefinition)
                .values(key=key, **row)
                .on_conflict_do_update(
                    index_elements=["key"],
                    set_={**row, "updated_at": db.func.now()},
                )
            )
            await session.commit()

    async def get_definition(self, key: str) -> SettingSpec | None:
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(db.SettingDefinition).where(db.SettingDefinition.key == key)
                )
            ).scalar_one_or_none()
        return SettingSpec.from_row(row) if row is not None else None

    async def list_definitions(self, category: str | None = None) -> list[SettingSpec]:
        async with self.session_factory() as session:
            stmt = select(db.SettingDefinition).order_by(
                db.SettingDefinition.category, db.SettingDefinition.sort_order,
                db.SettingDefinition.key,
            )
            if category:
                stmt = stmt.where(db.SettingDefinition.category == category)
            rows = (await session.execute(stmt)).scalars().all()
        return [SettingSpec.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    async def get_value(
        self, key: str, *, scope: SettingScope | str, scope_ref: str | None = None
    ) -> Any | None:
        """Return the current value at one explicit scope, or None if unset."""
        scope = str(scope)
        async with self.session_factory() as session:
            stmt = select(db.SettingValue.value).where(
                db.SettingValue.key == key,
                db.SettingValue.scope == scope,
            )
            stmt = _scope_ref_clause(stmt, scope_ref)
            stored = (await session.execute(stmt)).scalar_one_or_none()
        if stored is None:
            return None
        from services.settings.schema import _unwrap

        return _unwrap(stored)

    async def set_value(
        self,
        key: str,
        value: Any,
        *,
        scope: SettingScope | str,
        scope_ref: str | None = None,
        changed_by: str,
        reason: str | None = None,
    ) -> int:
        """Validate and persist one value; append history; return the new version.

        Raises ``HardRuleViolation`` / ``SettingValueError`` (and writes nothing)
        if the key is unknown, the value breaches a hard rule, or it fails schema
        validation. The definition's *default* is NOT used as a fallback scope —
        a caller asks for an explicit scope and gets exactly that.
        """
        scope = str(scope)
        spec = await self.get_definition(key)
        if spec is None:
            raise SettingValueError(f"unknown setting key: {key!r} (not in catalog)")

        # Hard rules are checked against the *coerced* value so a string 'true'
        # or a normalized enum cannot slip past a guard written for the real type.
        coerced = validate(spec, value)
        hard_rules.assert_not_violated(key, coerced)

        async with self.session_factory() as session:
            previous = await self._read_current(session, key, scope, scope_ref)
            version = await self._next_version(session, key, scope, scope_ref)

            session.add(
                db.SettingHistory(
                    key=key,
                    version=version,
                    scope=scope,
                    scope_ref=scope_ref,
                    previous_value=_wrap(previous) if previous is not None else None,
                    new_value=_wrap(coerced),
                    changed_by=changed_by,
                    reason=reason,
                )
            )
            await session.execute(
                pg_insert(db.SettingValue)
                .values(
                    key=key,
                    scope=scope,
                    scope_ref=scope_ref,
                    value=_wrap(coerced),
                    updated_by=changed_by,
                )
                .on_conflict_do_update(
                    # The conflict target is the COALESCE expression index, so an
                    # upsert of a global row (scope_ref NULL) updates in place
                    # rather than inserting a second NULL-scoped row (R12).
                    index_elements=[
                        db.SettingValue.key,
                        db.SettingValue.scope,
                        text("COALESCE(scope_ref, '')"),
                    ],
                    set_={
                        "value": _wrap(coerced),
                        "updated_by": changed_by,
                        "updated_at": db.func.now(),
                    },
                )
            )
            await session.commit()

        await self._audit(key, scope, scope_ref, coerced, changed_by, reason)
        logger.info(
            "settings.value_set",
            key=key,
            scope=scope,
            version=version,
            changed_by=changed_by,
        )
        return version

    async def list_values(
        self, key: str | None = None, scope: str | None = None
    ) -> list[dict[str, Any]]:
        """List current values (unwrapped), optionally filtered."""
        async with self.session_factory() as session:
            stmt = select(db.SettingValue).order_by(
                db.SettingValue.key, db.SettingValue.scope
            )
            if key:
                stmt = stmt.where(db.SettingValue.key == key)
            if scope:
                stmt = stmt.where(db.SettingValue.scope == scope)
            rows = (await session.execute(stmt)).scalars().all()
        from services.settings.schema import _unwrap

        return [
            {
                "key": r.key,
                "scope": r.scope,
                "scope_ref": r.scope_ref,
                "value": _unwrap(r.value),
                "updated_by": r.updated_by,
            }
            for r in rows
        ]

    async def delete_value(
        self,
        key: str,
        *,
        scope: SettingScope | str,
        scope_ref: str | None = None,
        changed_by: str,
        reason: str | None = None,
    ) -> bool:
        """Remove a scoped override (history is kept; this is not an erase).

        Returns True if a row was deleted. The next read falls back to a lower
        scope or the default (resolution is Phase 1b).
        """
        scope = str(scope)
        previous = None
        async with self.session_factory() as session:
            previous = await self._read_current(session, key, scope, scope_ref)
            if previous is None:
                return False
            version = await self._next_version(session, key, scope, scope_ref)
            session.add(
                db.SettingHistory(
                    key=key,
                    version=version,
                    scope=scope,
                    scope_ref=scope_ref,
                    previous_value=_wrap(previous),
                    # new_value is NOT NULL; a deletion records an explicit tombstone.
                    new_value={"deleted": True},
                    changed_by=changed_by,
                    reason=reason or "deleted override",
                )
            )
            stmt = db.SettingValue.__table__.delete().where(
                db.SettingValue.key == key,
                db.SettingValue.scope == scope,
            )
            stmt = _scope_ref_clause(stmt, scope_ref)
            result = await session.execute(stmt)
            await session.commit()
        await self._audit(key, scope, scope_ref, None, changed_by, reason or "deleted")
        return int(result.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _read_current(
        self, session, key: str, scope: str, scope_ref: str | None
    ) -> Any | None:
        stmt = select(db.SettingValue.value).where(
            db.SettingValue.key == key, db.SettingValue.scope == scope
        )
        stmt = _scope_ref_clause(stmt, scope_ref)
        stored = (await session.execute(stmt)).scalar_one_or_none()
        if stored is None:
            return None
        from services.settings.schema import _unwrap

        return _unwrap(stored)

    async def _next_version(
        self, session, key: str, scope: str, scope_ref: str | None
    ) -> int:
        stmt = select(db.func.coalesce(db.func.max(db.SettingHistory.version), 0)).where(
            db.SettingHistory.key == key, db.SettingHistory.scope == scope
        )
        stmt = _scope_ref_clause(stmt, scope_ref, table=db.SettingHistory)
        current = (await session.execute(stmt)).scalar_one()
        return int(current) + 1

    async def _audit(
        self,
        key: str,
        scope: str,
        scope_ref: str | None,
        value: Any,
        changed_by: str,
        reason: str | None,
    ) -> None:
        """Record the change in audit_logs via the governance auditor (best-effort).

        A sensitive key's value is masked — HR-NO-CREDENTIAL-LEAK: the trail shows
        that a setting changed and to what shape, never a secret value.
        """
        if self.auditor is None:
            return
        try:
            spec = await self.get_definition(key)
            shown = "<masked>" if (spec and spec.is_sensitive) else value
            await self.auditor.audit(
                module="settings",
                scope=f"settings:{scope}",
                pattern=key,
                verdict="set",
                reason=f"by={changed_by} ref={scope_ref} value={shown!r} reason={reason}",
            )
        except Exception:  # noqa: BLE001 — audit failure must not fail the write
            logger.warning("settings.audit_failed", key=key, exc_info=True)


def _scope_ref_clause(stmt, scope_ref: str | None, *, table=db.SettingValue):
    """Match a scope_ref exactly, treating None as SQL NULL (not equality)."""
    if scope_ref is None:
        return stmt.where(table.scope_ref.is_(None))
    return stmt.where(table.scope_ref == scope_ref)
