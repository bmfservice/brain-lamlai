"""Settings snapshot service — frozen immutable runtime views (Brain Second Phase 1b).

Creates and retrieves immutable SettingsSnapshot records so tasks or sessions
maintain deterministic behavior throughout their lifecycle, even if underlying
settings change dynamically mid-execution.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from core import database as db
from core.logging import get_logger
from services.settings.catalog import SPECS_BY_KEY
from services.settings.resolver import SettingsResolver

logger = get_logger(__name__)

# Key aliases mapping for convenient domain lookups
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "quality_gate_critical_failure_blocks": (
        "critical_failure_blocks_regardless_of_score",
        "quality_gate_critical_failure_blocks",
    ),
    "critical_failure_blocks_regardless_of_score": (
        "quality_gate_critical_failure_blocks",
        "critical_failure_blocks_regardless_of_score",
    ),
    "quality_gate_min_confidence": (
        "min_verification_confidence",
        "quality_gate_min_confidence",
    ),
    "min_verification_confidence": (
        "quality_gate_min_confidence",
        "min_verification_confidence",
    ),
    "quality_gate_min_score": (
        "min_overall_quality_score",
        "quality_gate_min_score",
    ),
    "min_overall_quality_score": (
        "quality_gate_min_score",
        "min_overall_quality_score",
    ),
    "max_retry_count": (
        "max_repair_attempts",
        "max_retry_count",
    ),
    "max_repair_attempts": (
        "max_retry_count",
        "max_repair_attempts",
    ),
}


class SettingsSnapshot:
    """Immutable snapshot of resolved settings for a task or session."""

    def __init__(
        self,
        values: Mapping[str, Any],
        resolved_from: Mapping[str, str] | None = None,
        *,
        snapshot_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.snapshot_id = snapshot_id or str(uuid.uuid4())
        self.task_id = task_id
        self.session_id = session_id
        self._values: dict[str, Any] = dict(values)
        self._resolved_from: dict[str, str] = dict(resolved_from or {})
        self.created_at = created_at or datetime.now(UTC)

    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value by key, with automatic alias lookup."""
        if key in self._values:
            return self._values[key]

        # Check aliases
        if key in _KEY_ALIASES:
            for alias in _KEY_ALIASES[key]:
                if alias in self._values:
                    return self._values[alias]

        # Check default from catalog if known
        if key in SPECS_BY_KEY:
            return SPECS_BY_KEY[key].default_value

        return default

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in self._values and key not in SPECS_BY_KEY:
            raise KeyError(key)
        return val

    def __contains__(self, key: str) -> bool:
        if key in self._values or key in SPECS_BY_KEY:
            return True
        if key in _KEY_ALIASES:
            return any(alias in self._values for alias in _KEY_ALIASES[key])
        return False

    @property
    def values(self) -> dict[str, Any]:
        """Return a copy of all resolved values."""
        return dict(self._values)

    @property
    def resolved_from(self) -> dict[str, str]:
        """Return a copy of resolved_from map (scope that won each key)."""
        return dict(self._resolved_from)

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "values": self._values,
            "resolved_from": self._resolved_from,
            "created_at": self.created_at.isoformat(),
        }

    def to_row(self) -> dict[str, Any]:
        """Render for insertion into database `settings_snapshots` table."""
        return {
            "id": uuid.UUID(self.snapshot_id),
            "task_id": uuid.UUID(self.task_id) if self.task_id else None,
            "session_id": uuid.UUID(self.session_id) if self.session_id else None,
            "values": self._values,
            "resolved_from": self._resolved_from,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> SettingsSnapshot:
        """Reconstruct snapshot from DB row."""
        return cls(
            values=row.values,
            resolved_from=row.resolved_from or {},
            snapshot_id=str(row.id),
            task_id=str(row.task_id) if row.task_id else None,
            session_id=str(row.session_id) if row.session_id else None,
            created_at=row.created_at,
        )


def create_default_snapshot(
    task_id: str | None = None,
    session_id: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SettingsSnapshot:
    """Create a snapshot from catalog default values and optional overrides."""
    resolved_values: dict[str, Any] = {}
    resolved_from: dict[str, str] = {}

    for spec in SPECS_BY_KEY.values():
        resolved_values[spec.key] = spec.default_value
        resolved_from[spec.key] = "default"

    if overrides:
        for k, v in overrides.items():
            resolved_values[k] = v
            resolved_from[k] = "override"

    return SettingsSnapshot(
        values=resolved_values,
        resolved_from=resolved_from,
        task_id=task_id,
        session_id=session_id,
    )


async def resolve_settings_snapshot(
    store: Any = None,
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    task_overrides: Mapping[str, Any] | None = None,
    user_overrides: Mapping[str, Any] | None = None,
) -> SettingsSnapshot:
    """Resolve full settings snapshot respecting the 7-level hierarchy."""
    resolved_values: dict[str, Any] = {}
    resolved_from: dict[str, str] = {}

    for spec in SPECS_BY_KEY.values():
        resolved_values[spec.key] = spec.default_value
        resolved_from[spec.key] = "default"

    if store is not None:
        try:
            resolver = SettingsResolver(store.session_factory, store=store)
            vals, scopes = await resolver.resolve_all(task_id=task_id, session_id=session_id)
            resolved_values.update(vals)
            resolved_from.update(scopes)
        except Exception:
            pass

    if user_overrides:
        for k, v in user_overrides.items():
            resolved_values[k] = v
            resolved_from[k] = "user_preference"

    if task_overrides:
        for k, v in task_overrides.items():
            resolved_values[k] = v
            resolved_from[k] = "task"

    return SettingsSnapshot(
        values=resolved_values,
        resolved_from=resolved_from,
        task_id=task_id,
        session_id=session_id,
    )


class SettingsSnapshotService:
    """Service to create and query frozen immutable settings snapshots."""

    def __init__(self, session_factory, resolver: SettingsResolver) -> None:
        self.session_factory = session_factory
        self.resolver = resolver

    async def create_snapshot(
        self,
        *,
        task_id: uuid.UUID | str | None = None,
        session_id: uuid.UUID | str | None = None,
    ) -> uuid.UUID:
        """Create an immutable snapshot of all resolved settings and return snapshot_id."""
        task_uuid = (
            uuid.UUID(str(task_id))
            if task_id is not None and not isinstance(task_id, uuid.UUID)
            else task_id
        )
        session_uuid = (
            uuid.UUID(str(session_id))
            if session_id is not None and not isinstance(session_id, uuid.UUID)
            else session_id
        )

        values, resolved_from = await self.resolver.resolve_all(
            task_id=str(task_id) if task_id is not None else None,
            session_id=str(session_id) if session_id is not None else None,
        )

        snapshot_id = uuid.uuid4()
        async with self.session_factory() as session:
            snapshot = db.SettingsSnapshot(
                id=snapshot_id,
                task_id=task_uuid,
                session_id=session_uuid,
                values=values,
                resolved_from=resolved_from,
            )
            session.add(snapshot)
            await session.commit()

        logger.info(
            "settings.snapshot_created",
            snapshot_id=str(snapshot_id),
            task_id=str(task_id) if task_id else None,
            session_id=str(session_id) if session_id else None,
            key_count=len(values),
        )
        return snapshot_id

    async def get_snapshot(self, snapshot_id: uuid.UUID | str) -> dict[str, Any] | None:
        """Read frozen values for a snapshot. Returns None if snapshot does not exist."""
        snap_uuid = (
            uuid.UUID(str(snapshot_id))
            if not isinstance(snapshot_id, uuid.UUID)
            else snapshot_id
        )
        async with self.session_factory() as session:
            stmt = select(db.SettingsSnapshot).where(db.SettingsSnapshot.id == snap_uuid)
            snapshot = (await session.execute(stmt)).scalar_one_or_none()

        if snapshot is None:
            return None

        return dict(snapshot.values)

    async def get_snapshot_record(
        self, snapshot_id: uuid.UUID | str
    ) -> dict[str, Any] | None:
        """Read full snapshot record including metadata."""
        snap_uuid = (
            uuid.UUID(str(snapshot_id))
            if not isinstance(snapshot_id, uuid.UUID)
            else snapshot_id
        )
        async with self.session_factory() as session:
            stmt = select(db.SettingsSnapshot).where(db.SettingsSnapshot.id == snap_uuid)
            snapshot = (await session.execute(stmt)).scalar_one_or_none()

        if snapshot is None:
            return None

        return {
            "id": snapshot.id,
            "task_id": snapshot.task_id,
            "session_id": snapshot.session_id,
            "values": snapshot.values,
            "resolved_from": snapshot.resolved_from,
            "created_at": snapshot.created_at,
        }
