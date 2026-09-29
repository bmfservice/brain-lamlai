"""Setting change history: versioning, diff, rollback (Brain Second Phase 1a).

Every value write appends a ``setting_history`` row (done by ``SettingsStore``).
This module reads that trail and implements ``rollback`` — which is *not* a
delete: rolling back to version N reads that version's value and writes it again
through the store, producing a NEW version. The trail therefore never loses a
step, which is what makes the change history auditable rather than merely
current-state.

``diff`` compares two versions of one key; ``list_history`` returns the trail for
a key (newest first). Reads return unwrapped values for callers/UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from core import database as db
from core.logging import get_logger
from services.settings.schema import SettingScope, _unwrap

logger = get_logger(__name__)


@dataclass(frozen=True)
class HistoryEntry:
    """One recorded change to a setting."""

    version: int
    key: str
    scope: str
    scope_ref: str | None
    previous_value: Any
    new_value: Any
    changed_at: Any
    changed_by: str
    reason: str | None


class SettingsHistory:
    """Read the version trail and roll back through it."""

    def __init__(self, session_factory, store) -> None:
        self.session_factory = session_factory
        # Rollback delegates to the store so it gets the same validation,
        # hard-rule check and history-append as any other write.
        self.store = store

    async def list_history(
        self,
        key: str,
        *,
        scope: SettingScope | str | None = None,
        scope_ref: str | None = None,
        limit: int = 50,
    ) -> list[HistoryEntry]:
        """Return a key's change trail, newest version first."""
        async with self.session_factory() as session:
            stmt = (
                select(db.SettingHistory)
                .where(db.SettingHistory.key == key)
                .order_by(db.SettingHistory.version.desc())
                .limit(limit)
            )
            if scope is not None:
                stmt = stmt.where(db.SettingHistory.scope == str(scope))
                stmt = _ref_clause(stmt, scope_ref)
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_entry(r) for r in rows]

    async def get_version(
        self, key: str, version: int, *, scope: str, scope_ref: str | None = None
    ) -> HistoryEntry | None:
        async with self.session_factory() as session:
            stmt = select(db.SettingHistory).where(
                db.SettingHistory.key == key,
                db.SettingHistory.version == version,
                db.SettingHistory.scope == scope,
            )
            stmt = _ref_clause(stmt, scope_ref)
            row = (await session.execute(stmt)).scalar_one_or_none()
        return _to_entry(row) if row is not None else None

    async def diff(
        self,
        key: str,
        version_from: int,
        version_to: int,
        *,
        scope: str,
        scope_ref: str | None = None,
    ) -> dict[str, Any]:
        """Return the from/to values (and metadata) between two versions."""
        a = await self.get_version(key, version_from, scope=scope, scope_ref=scope_ref)
        b = await self.get_version(key, version_to, scope=scope, scope_ref=scope_ref)
        if a is None or b is None:
            raise KeyError(f"version {version_from} or {version_to} not found for {key}")
        return {
            "key": key,
            "scope": scope,
            "from_version": a.version,
            "to_version": b.version,
            "from_value": a.new_value,
            "to_value": b.new_value,
            "changed_by": b.changed_by,
            "reason": b.reason,
            "changed_at": b.changed_at,
        }

    async def rollback(
        self,
        key: str,
        to_version: int,
        *,
        scope: SettingScope | str,
        scope_ref: str | None = None,
        changed_by: str,
        reason: str | None = None,
    ) -> int:
        """Restore the value recorded at ``to_version`` as a NEW version.

        Returns the new version number. Raises ``KeyError`` if the target version
        does not exist. A tombstone (``{"deleted": True}``) is not a restorable
        value — rolling back to one is refused, since the caller almost certainly
        means the last real value before the delete.
        """
        scope = str(scope)
        entry = await self.get_version(key, to_version, scope=scope, scope_ref=scope_ref)
        if entry is None:
            raise KeyError(f"version {to_version} not found for {key} in scope {scope}")
        target = entry.new_value
        if isinstance(target, dict) and target.get("deleted") is True:
            raise ValueError(
                f"version {to_version} of {key} is a deletion tombstone; "
                "roll back to an earlier real value instead"
            )
        return await self.store.set_value(
            key,
            target,
            scope=scope,
            scope_ref=scope_ref,
            changed_by=changed_by,
            reason=reason or f"rollback to v{to_version}",
        )


def _to_entry(row: db.SettingHistory) -> HistoryEntry:
    return HistoryEntry(
        version=int(row.version),
        key=row.key,
        scope=row.scope,
        scope_ref=row.scope_ref,
        previous_value=_unwrap(row.previous_value) if row.previous_value is not None else None,
        new_value=_unwrap(row.new_value),
        changed_at=row.changed_at,
        changed_by=row.changed_by,
        reason=row.reason,
    )


def _ref_clause(stmt, scope_ref: str | None):
    if scope_ref is None:
        return stmt.where(db.SettingHistory.scope_ref.is_(None))
    return stmt.where(db.SettingHistory.scope_ref == scope_ref)
